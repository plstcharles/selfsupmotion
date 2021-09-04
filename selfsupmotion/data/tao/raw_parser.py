import glob
import logging
import os
import typing
import zipfile

import numpy as np

import selfsupmotion.data.utils as data_utils
import selfsupmotion.data.tao.utils as tao_utils

logger = logging.getLogger(__name__)


class TAORawVideoParser(tao_utils.TAOVideoParserBase):

    def __init__(
        self,
        data_root_path: typing.AnyStr,
        video_info: typing.Dict,
        dataset_info: typing.Dict,
        dataset_zip_fd: zipfile.ZipFile,
        convert_to_rgb: bool,
    ):
        super().__init__(
            id=video_info["id"],
            name=video_info["name"],
            metadata=video_info["metadata"],
            width=video_info["width"],
            height=video_info["height"],
            neg_category_ids=video_info["neg_category_ids"],
            not_exhaustive_category_ids=video_info["not_exhaustive_category_ids"],
        )

        # next, assign the reverse-lookup maps to fetch annotations/tracks/images based on their IDs
        self.categories_info = {cid: dataset_info["categories"][cid] for cid in self.not_exhaustive_category_ids}
        self.annotations_info = {info["id"]: info for info in dataset_info["annotations_map"][self.id]}
        self.tracks_info = {info["id"]: info for info in dataset_info["tracks_map"][self.id]}
        self.images_info = {info["id"]: info for info in dataset_info["images_map"][self.id]}

        # finally, determine how to load the data, how many frames there are, and prepare the metadata list
        self.can_load_images_directly = tao_utils.check_if_images_are_on_disk(
            data_root_path=data_root_path,
            rel_dir_path=self.name,
            file_names=[iinfo["file_name"] for iinfo in self.images_info.values()],
        )
        self.dataset_zip_fd = dataset_zip_fd  # kept in case we need to fetch images from zip
        self.convert_to_rgb = convert_to_rgb
        self.frames_metadata = self._generate_frame_metadata(data_root_path=data_root_path)

    def _generate_frame_metadata(self, data_root_path: typing.AnyStr):
        # first, let's get the total frame count for this video as well as the path to all frames
        dataset_content = self.dataset_zip_fd.namelist()  # should be fast once zip is opened
        video_prefix = os.path.join("frames", self.name) + "/"
        in_zip_frame_paths = sorted([
            p for p in dataset_content if p.startswith(video_prefix) and p.endswith(".jpg")
        ])
        if self.can_load_images_directly:
            on_disk_frame_dir_path = os.path.join(data_root_path, "frames", self.name)
            on_disk_frame_paths = sorted(glob.glob(on_disk_frame_dir_path + "/*.jpg"))
            assert len(on_disk_frame_paths) == len(in_zip_frame_paths)
            frame_paths = on_disk_frame_paths  # should load faster...? right?
        else:
            frame_paths = in_zip_frame_paths  # and this one slower? (yep: 75% slower on HDD)
        assert len(frame_paths) > 0

        # next, update the 'frame_index' stored in the image metadata to 0-based
        for iid, iinfo in self.images_info.items():
            assert iinfo["video_id"] == self.id and iinfo["video"] == self.name
            assert iinfo["width"] == self.width and iinfo["height"] == self.height
            real_frame_idx = in_zip_frame_paths.index("frames/" + iinfo["file_name"])
            del iinfo["frame_index"]  # the original frame index is useless to us and mistake-prone
            iinfo["frame_idx"] = real_frame_idx  # this is how we'll map real frame indices below

        # next, we need a way to map real frame indices to corresponding annotations, so let's create a temp map
        frame_idx_to_annotation_ids_map = {idx: [] for idx in range(len(frame_paths))}
        for aid, annot in self.annotations_info.items():
            image_info = self.images_info[annot["image_id"]]
            real_frame_idx = image_info["frame_idx"]
            frame_idx_to_annotation_ids_map[real_frame_idx].append(aid)

        # we'll create the map of metadata that contains all relevant info for each frame
        frames_metadata = [  # initialize basic metadata w/ empty tracks map
            {
                "image_data": None,  # will only be filled at runtime
                "image_id": None,  # will be filled below if there is a match
                "image_path": frame_paths[idx],
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
            } for idx in range(len(frame_paths))
        ]

        # finally, re-run for each frame and fill in with the available info based on all maps
        track_latest_updates = {track_id: None for track_id in self.tracks_info}
        target_annot_keys = ["segmentation", "bbox", "area", "iscrowd"]
        for frame_idx in range(len(frames_metadata)):
            curr_frame_annot_ids = frame_idx_to_annotation_ids_map[frame_idx]
            if curr_frame_annot_ids:
                matched_image_ids = np.unique([
                    self.annotations_info[aid]["image_id"] for aid in curr_frame_annot_ids
                ])
                assert len(matched_image_ids) == 1
                frames_metadata[frame_idx]["image_id"] = int(matched_image_ids[0])
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

    def _load_frame_data(self, frame_data: typing.Dict):
        frame_path = frame_data["image_path"]
        if self.can_load_images_directly:
            with open(frame_path, "rb") as fd:
                image_data = data_utils.decode_jpeg(
                    data=fd.read(),
                    convert_to_rgb=self.convert_to_rgb,
                )
        else:
            image_data = data_utils.decode_jpeg(
                data=self.dataset_zip_fd.read(frame_path),
                convert_to_rgb=self.convert_to_rgb,
            )
        assert image_data.shape == (self.height, self.width, 3)
        frame_data["image_data"] = image_data
