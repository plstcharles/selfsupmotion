# this module will export TAO video tracks to HDF5 archives that can be randomly accessed
# (note that the export framerate is a parameter that must be controlled below)
import copy
import glob
import itertools
import json
import logging
import os
import typing
import zipfile

import h5py
import numpy as np
import tqdm

import selfsupmotion.data.utils as utils

logger = logging.getLogger(__name__)


private_dataset_names = ["AVA", "HACS"]


def _get_parent_zip_name_from_prefix(prefix: typing.AnyStr):
    if prefix.startswith("train"):
        return "1-TAO_TRAIN.zip"
    if prefix.startswith("valid"):
        return "2-TAO_VAL.zip"
    if prefix.startswith("test"):
        return "3-TAO_TEST.zip"
    raise NotImplementedError


def _get_annot_file_name_from_prefix(prefix: typing.AnyStr):
    if prefix.startswith("train"):
        return "train_with_freeform.json"
    if prefix.startswith("valid"):
        return "validation_with_freeform.json"
    if prefix.startswith("test"):
        return "test_without_annotations.json"
    raise NotImplementedError


class TAOVideoParser:

    def __init__(
        self,
        data_root_path: typing.AnyStr,
        video_info: typing.Dict,
        dataset_info: typing.Dict,
        dataset_zip_fd: zipfile.ZipFile,
    ):
        # first, extract high-level info about the sequence itself
        self.id = video_info["id"]  # unique identifier (integer)
        self.name = video_info["name"]  # this 'name' is actually the relative path to the video folder
        self.metadata = video_info["metadata"]  # this contains mostly useless info (dataset name, user name)
        self.width = video_info["width"]  # should be constant across the entire video sequence
        self.height = video_info["height"]  # should be constant across the entire video sequence
        self.neg_category_ids = video_info["neg_category_ids"]  # ids of categories IGNORED in this sequence
        self.not_exhaustive_category_ids = video_info["not_exhaustive_category_ids"]  # ids of annotated cats
        # note: the 'non exhaustive' list means that not all tracked objects might have a known category
        self.categories_info = {cid: dataset_info["categories"][cid] for cid in self.not_exhaustive_category_ids}
        self.data_root_path = data_root_path

        # next, create some convenient reverse-lookup maps to fetch annotations/tracks/images based on their IDs
        self.annotations_info = {info["id"]: info for info in dataset_info["annotations_map"][self.id]}
        self.tracks_info = {info["id"]: info for info in dataset_info["tracks_map"][self.id]}
        self.images_info = {info["id"]: info for info in dataset_info["images_map"][self.id]}

        # finally, determine how to load the data, how many frames there are, and prepare the metadata list
        self.can_load_images_directly = self._check_if_images_are_on_disk()
        self.dataset_zip_fd = dataset_zip_fd  # kept in case we need to fetch images from zip
        self.frame_count = None  # will be determined when generating frame metadata below
        self.frame_paths = []  # will be filled when generating frame metadata below
        self.frames_metadata = self._generate_frame_metadata()

    def _check_if_images_are_on_disk(self):
        frames_root_path = os.path.join(self.data_root_path, "frames")
        frames_dir_path = os.path.join(frames_root_path, self.name)
        if not os.path.isdir(frames_dir_path):
            return False
        image_paths = [os.path.join(frames_root_path, iinfo["file_name"]) for iinfo in self.images_info.values()]
        return len(image_paths) > 0 and all([os.path.isfile(p) for p in image_paths])

    def _generate_frame_metadata(self):
        # first, let's get the total frame count for this video as well as the path to all frames
        dataset_content = self.dataset_zip_fd.namelist()  # should be fast once zip is opened
        video_prefix = os.path.join("frames", self.name) + "/"
        in_zip_frame_paths = sorted([
            p for p in dataset_content if p.startswith(video_prefix) and p.endswith(".jpg")
        ])
        if self.can_load_images_directly:
            on_disk_frame_dir_path = os.path.join(self.data_root_path, "frames", self.name)
            on_disk_frame_paths = sorted(glob.glob(on_disk_frame_dir_path + "/*.jpg"))
            assert len(on_disk_frame_paths) == len(in_zip_frame_paths)
            self.frame_paths = on_disk_frame_paths  # should load faster...? right?
        else:
            self.frame_paths = in_zip_frame_paths  # and this one slower? (yep: 75% slower on HDD)
        self.frame_count = len(self.frame_paths)
        assert self.frame_count > 0

        # next, update the 'frame_index' stored in the image metadata to 0-based
        for iid, iinfo in self.images_info.items():
            assert iinfo["video_id"] == self.id and iinfo["video"] == self.name
            assert iinfo["width"] == self.width and iinfo["height"] == self.height
            real_frame_idx = in_zip_frame_paths.index("frames/" + iinfo["file_name"])
            del iinfo["frame_index"]  # the original frame index is useless to us and mistake-prone
            iinfo["frame_idx"] = real_frame_idx  # this is how we'll map real frame indices below

        # next, we need a way to map real frame indices to corresponding annotations, so let's create a temp map
        frame_idx_to_annotation_ids_map = {idx: [] for idx in range(self.frame_count)}
        for aid, annot in self.annotations_info.items():
            image_info = self.images_info[annot["image_id"]]
            real_frame_idx = image_info["frame_idx"]
            frame_idx_to_annotation_ids_map[real_frame_idx].append(aid)

        # we'll create the map of metadata that contains all relevant info for each frame
        frames_metadata = [  # initialize basic metadata w/ empty tracks map
            {
                "image_data": None,  # will only be filled at runtime
                "image_id": None,  # will be filled below if there is a match
                "frame_idx": idx,
                "tracks": {
                    tid: {
                        # some categories do not have associated info; sad, but it's not world-ending
                        "category_id": self.tracks_info[tid]["category_id"],
                        # we'll provide the frame index of the previous annotation
                        "last_annotation_frame_idx": None,
                        # this is the field that will be filled if we have a new annotation for this track
                        "annotation": None,
                    } for tid in self.tracks_info
                },
            } for idx in range(self.frame_count)
        ]

        # finally, re-run for each frame and fill in with the available info based on all maps
        track_latest_updates = {track_id: None for track_id in self.tracks_info}
        target_annot_keys = ["segmentation", "bbox", "area", "iscrowd"]
        for frame_idx in range(self.frame_count):
            curr_frame_annot_ids = frame_idx_to_annotation_ids_map[frame_idx]
            if curr_frame_annot_ids:
                matched_image_ids = np.unique([
                    self.annotations_info[aid]["image_id"] for aid in curr_frame_annot_ids
                ])
                assert len(matched_image_ids) == 1
                frames_metadata[frame_idx]["image_id"] = matched_image_ids[0]
                assert self.images_info[matched_image_ids[0]]["frame_idx"] == frame_idx
                for aid in curr_frame_annot_ids:
                    annot_info = self.annotations_info[aid]
                    track_id = annot_info["track_id"]
                    frames_metadata[frame_idx]["tracks"][track_id]["annotation"] = {
                        key: annot_info[key] for key in target_annot_keys
                    }
                    last_update = track_latest_updates[track_id]
                    assert last_update is None or last_update < frame_idx
                    track_latest_updates[track_id] = frame_idx

        return frames_metadata

    def __len__(self) -> int:
        return self.frame_count

    def __getitem__(self, frame_idx: int) -> typing.Dict:
        if frame_idx < 0:
            frame_idx = self.frame_count - frame_idx
        assert 0 <= frame_idx < self.frame_count
        # the metadata for the frame is all ready, we just need to load the image data itself
        frame_data = copy.deepcopy(self.frames_metadata[frame_idx])
        frame_path = self.frame_paths[frame_idx]
        if self.can_load_images_directly:
            with open(frame_path, "rb") as fd:
                image_data = utils.decode_jpeg(fd.read())
        else:
            image_data = utils.decode_jpeg(self.dataset_zip_fd.read(frame_path))
        assert image_data.shape == (self.height, self.width, 3)
        frame_data["image_data"] = image_data
        return frame_data


class TAODataModule:

    def __init__(
        self,
        data_root_path: typing.AnyStr,
        skip_private_datasets: bool,
    ):
        self.data_root_path = data_root_path
        assert os.path.isdir(data_root_path)
        logger.info("Parsing training split...")
        self.train_video_parsers = self._create_video_parsers(
            data_root_path=self.data_root_path,
            prefix="train",
            skip_private_datasets=skip_private_datasets,
        )
        logger.info("Parsing validation split...")
        self.valid_video_parsers = self._create_video_parsers(
            data_root_path=self.data_root_path,
            prefix="valid",
            skip_private_datasets=skip_private_datasets,
        )
        logger.info("Parsing test split...")
        self.test_video_parsers = self._create_video_parsers(
            data_root_path=self.data_root_path,
            prefix="test",
            skip_private_datasets=skip_private_datasets,
        )
        logger.info("Verifying split overlap...")
        self._verify_split_overlap((
            self.train_video_parsers,
            self.valid_video_parsers,
            self.test_video_parsers,
        ))
        logger.info("Data module ready.")

    @staticmethod
    def _create_video_parsers(
        data_root_path: typing.AnyStr,
        prefix: typing.AnyStr,
        skip_private_datasets: bool,
    ):
        # first, open the JSON file that contains all the annotations for the specified split
        annot_root_path = os.path.join(data_root_path, "annotations")
        assert os.path.isdir(annot_root_path)
        annot_file_path = os.path.join(annot_root_path, _get_annot_file_name_from_prefix(prefix))
        logger.info(f"\tLoading annotations from: {annot_file_path}")
        assert os.path.isfile(annot_file_path)
        with open(annot_file_path, "r") as fd:
            dataset_info = json.load(fd)
        logger.info("\tGenerating reverse lookup maps...")
        TAODataModule._add_reverse_lookup_maps_to_dataset_info(dataset_info)

        # next, open the zip & get ready to parse all the metadata from the JSON+zip simultaneously
        video_parsers = []
        zip_path = os.path.join(data_root_path, _get_parent_zip_name_from_prefix(prefix))
        assert zipfile.is_zipfile(zip_path)
        logger.info(f"\tParsing content for zip file: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zfd:
            logger.info(f"\tCreating video parsers...")
            iterator = tqdm.tqdm(dataset_info["videos"], desc="Fetching video metadata")
            for video_info in iterator:
                if skip_private_datasets and video_info["metadata"]["dataset"] in private_dataset_names:
                    continue
                parser = TAOVideoParser(
                    data_root_path=data_root_path,
                    video_info=video_info,
                    dataset_info=dataset_info,
                    dataset_zip_fd=zfd,
                )
                assert len(parser) > 0
                if len(parser.annotations_info) == 0 and prefix in ["train", "valid"]:
                    logger.warning(f"no annotation found for video: {video_info['name']}")
                    continue  # we skip these videos since they cannot be used for anything
                # import cv2 as cv
                # for frame_idx in range(len(parser)):
                #     frame_data = parser[frame_idx]
                #     got_gt = False
                #     for tid, track_info in frame_data["tracks"].items():
                #         if track_info["annotation"]:
                #             bbox = track_info["annotation"]["bbox"]
                #             cv.rectangle(
                #                 frame_data["image_data"],
                #                 pt1=(bbox[0], bbox[1]),
                #                 pt2=(bbox[0] + bbox[2], bbox[1] + bbox[3]),
                #                 color=(112, 112, 255),
                #                 thickness=1,
                #             )
                #             got_gt = True
                #     image = cv.cvtColor(frame_data["image_data"], cv.COLOR_RGB2BGR)
                #     image = cv.resize(image, dsize=(-1, -1), fx=4, fy=4, interpolation=cv.INTER_NEAREST)
                #     cv.imshow("test", image)
                #     cv.waitKey(0 if got_gt else 1)
                video_parsers.append(parser)
        assert len(np.unique([v.id for v in video_parsers])) == len(video_parsers)
        return video_parsers

    @staticmethod
    def _verify_split_overlap(parsers: typing.Iterable[typing.Sequence[TAOVideoParser]]):
        for combo in itertools.combinations(parsers, 2):
            ids = ([parser.id for parser in combo[i]] for i in range(2))
            assert len(np.intersect1d(*ids)) == 0

    @staticmethod
    def _get_empty_video_map(dataset_info: typing.Dict):
        return {vid["id"]: [] for vid in dataset_info["videos"]}

    @staticmethod
    def _add_reverse_lookup_maps_to_dataset_info(dataset_info: typing.Dict):
        annotations_map = TAODataModule._get_empty_video_map(dataset_info)
        for annot_info in dataset_info["annotations"]:
            annotations_map[annot_info["video_id"]].append(annot_info)
        dataset_info["annotations_map"] = annotations_map
        tracks_map = TAODataModule._get_empty_video_map(dataset_info)
        for track_info in dataset_info["tracks"]:
            tracks_map[track_info["video_id"]].append(track_info)
        dataset_info["tracks_map"] = tracks_map
        images_map = TAODataModule._get_empty_video_map(dataset_info)
        for image_info in dataset_info["images"]:
            images_map[image_info["video_id"]].append(image_info)
        dataset_info["images_map"] = images_map


def main():
    data_root_path = "/nfs/server/datasets/tao/"
    data_module = TAODataModule(
        data_root_path=data_root_path,
        skip_private_datasets=True,
    )
    # @@@@@@@@ TODO: package hdf5 & write base class for hdf5-parser and raw-parser


if __name__ == "__main__":
    # we want to highlight how much better representations are when using temporal-pairs instead of synthetic-aug-pairs

    # what % of the time is the model capable of finding the 'real' match among X candidates when the candidates come from Y frames in the future
    # draw this as a curve: X axis = for each pair (t, t+x) at different offsets of x, Y axis = average % of correct matches found across all seqs

    logging.basicConfig(level=logging.INFO)
    main()
    print("all done")
