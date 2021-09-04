import itertools
import json
import logging
import os
import typing
import zipfile

import numpy as np
import tqdm

import selfsupmotion.data.tao.utils as tao_utils
import selfsupmotion.data.tao.raw_parser as raw_parser

logger = logging.getLogger(__name__)


class TAODataModule:

    def __init__(
        self,
        data_root_path: typing.AnyStr,
        skip_private_datasets: bool,
        load_train: bool = True,
        load_valid: bool = True,
        load_test: bool = True,
        convert_to_rgb: bool = False,
    ):
        assert load_train or load_valid or load_test
        logger.info(f"Data module initializing at: {data_root_path}")
        self.data_root_path = data_root_path
        assert os.path.isdir(data_root_path)
        self.train_video_parsers = self._create_video_parsers(
            data_root_path=self.data_root_path,
            prefix="train",
            skip_private_datasets=skip_private_datasets,
            convert_to_rgb=convert_to_rgb,
        ) if load_train else []
        self.valid_video_parsers = self._create_video_parsers(
            data_root_path=self.data_root_path,
            prefix="valid",
            skip_private_datasets=skip_private_datasets,
            convert_to_rgb=convert_to_rgb,
        ) if load_valid else []
        self.test_video_parsers = self._create_video_parsers(
            data_root_path=self.data_root_path,
            prefix="test",
            skip_private_datasets=skip_private_datasets,
            convert_to_rgb=convert_to_rgb,
        ) if load_test else []
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
        convert_to_rgb: bool,
    ):
        logger.info(f"Parsing {prefix} split...")
        # first, open the JSON file that contains all the annotations for the specified split
        annot_root_path = os.path.join(data_root_path, "annotations")
        assert os.path.isdir(annot_root_path)
        annot_file_path = os.path.join(annot_root_path, tao_utils.get_annot_file_name_from_prefix(prefix))
        logger.info(f"\tLoading annotations from: {annot_file_path}")
        assert os.path.isfile(annot_file_path)
        with open(annot_file_path, "r") as fd:
            dataset_info = json.load(fd)
        logger.info("\tGenerating reverse lookup maps...")
        TAODataModule._add_reverse_lookup_maps_to_dataset_info(dataset_info)

        # next, open the zip & get ready to parse all the metadata from the JSON+zip simultaneously
        video_parsers = []
        zip_path = os.path.join(data_root_path, tao_utils.get_parent_zip_name_from_prefix(prefix))
        assert zipfile.is_zipfile(zip_path)
        logger.info(f"\tParsing content for zip file: {zip_path}")
        # note: zip will stay opened in memory until all parsers are destroyed!
        # (should not be big enough to cause issues though)
        zfd = zipfile.ZipFile(zip_path, "r")
        logger.info(f"\tCreating video parsers...")
        iterator = tqdm.tqdm(dataset_info["videos"], desc="Fetching video metadata")
        for vidx, video_info in enumerate(iterator): #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            if skip_private_datasets:
                if video_info["metadata"]["dataset"] in tao_utils.private_dataset_names:
                    continue
            if vidx > 10:  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                break
            parser = raw_parser.TAORawVideoParser(
                data_root_path=data_root_path,
                video_info=video_info,
                dataset_info=dataset_info,
                dataset_zip_fd=zfd,
                convert_to_rgb=convert_to_rgb,
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
            #     image = cv.resize(image, dsize=(-1, -1), fx=4, fy=4, interpolation=cv.INTER_NEAREST)
            #     cv.imshow("test", image)
            #     cv.waitKey(0 if got_gt else 1)
            video_parsers.append(parser)
        assert len(np.unique([v.id for v in video_parsers])) == len(video_parsers)
        return video_parsers

    @staticmethod
    def _verify_split_overlap(parsers: typing.Iterable[typing.Sequence[tao_utils.TAOVideoParserBase]]):
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
