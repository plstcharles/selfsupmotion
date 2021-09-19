import itertools
import json
import logging
import os
import typing
import zipfile

import h5py
import numpy as np
import tqdm

import selfsupmotion.data.tao.utils as tao_utils
import selfsupmotion.data.tao.raw_parser as raw_parser
import selfsupmotion.data.tao.hdf5_parser as hdf5_parser

logger = logging.getLogger(__name__)


class TAODataModule:
    def __init__(
        self,
        data_root_path: typing.AnyStr,
        use_hdf5_packages: bool = True,
        import_only_annot_frames: bool = True,
        skip_private_datasets: bool = True,
        load_train: bool = True,
        load_valid: bool = True,
        load_test: bool = True,
        convert_to_rgb: bool = False,
    ):
        assert load_train or load_valid or load_test
        logger.info(f"Data module initializing at: {data_root_path}")
        assert os.path.isdir(data_root_path)
        self.data_root_path = data_root_path
        self.use_hdf5_packages = use_hdf5_packages
        self.import_only_annot_frames = import_only_annot_frames
        self.skip_private_datasets = skip_private_datasets
        self.load_train, self.load_valid, self.load_test = (
            load_train,
            load_valid,
            load_test,
        )
        self.convert_to_rgb = convert_to_rgb
        self.train_parsers, self.valid_parsers, self.test_parsers = [], [], []
        self._reload_parsers()
        logger.info("Data module ready.")

    def _reload_parsers(self):
        self.train_parsers = (
            self._create_video_parsers(prefix="train") if self.load_train else []
        )
        self.valid_parsers = (
            self._create_video_parsers(prefix="valid") if self.load_valid else []
        )
        self.test_parsers = (
            self._create_video_parsers(prefix="test") if self.load_test else []
        )
        self._verify_split_overlap(
            (
                self.train_parsers,
                self.valid_parsers,
                self.test_parsers,
            )
        )

    def _create_video_parsers(self, prefix: typing.AnyStr):
        if self.import_only_annot_frames and prefix == "test":
            return []
        logger.info(f"Parsing {prefix} split...")
        if self.use_hdf5_packages:
            hdf5_file_name = tao_utils.get_hdf5_file_name_from_prefix(
                prefix, self.import_only_annot_frames
            )
            hdf5_file_path = os.path.join(self.data_root_path, hdf5_file_name)
            video_parsers = TAODataModule._create_hdf5_video_parsers(
                hdf5_file_path=hdf5_file_path,
                skip_private_datasets=self.skip_private_datasets,
                prefix=prefix,
                convert_to_rgb=self.convert_to_rgb,
            )
        else:
            annot_root_path = os.path.join(self.data_root_path, "annotations")
            assert os.path.isdir(annot_root_path)
            annot_file_path = os.path.join(
                annot_root_path, tao_utils.get_annot_file_name_from_prefix(prefix)
            )
            zip_file_path = os.path.join(
                self.data_root_path, tao_utils.get_parent_zip_name_from_prefix(prefix)
            )
            assert zipfile.is_zipfile(zip_file_path)
            video_parsers = TAODataModule._create_raw_video_parsers(
                data_root_path=self.data_root_path,
                annot_file_path=annot_file_path,
                zip_file_path=zip_file_path,
                import_only_annot_frames=self.import_only_annot_frames,
                skip_private_datasets=self.skip_private_datasets,
                prefix=prefix,
                convert_to_rgb=self.convert_to_rgb,
            )
        if not self.skip_private_datasets:
            assert any(
                [
                    any([k in p.name for k in tao_utils.private_dataset_names])
                    for p in video_parsers
                ]
            )
        assert len(np.unique([v.id for v in video_parsers])) == len(video_parsers)
        return video_parsers

    @staticmethod
    def _create_hdf5_video_parsers(
        hdf5_file_path: typing.AnyStr,
        skip_private_datasets: bool,
        prefix: typing.AnyStr,
        convert_to_rgb: bool,
    ) -> typing.Sequence[tao_utils.TAOVideoParserBase]:
        with h5py.File(hdf5_file_path) as h5fd:  # sneak-peek for video count
            hdf5_video_count = h5fd.attrs["video_count"]
        video_parsers = []
        iterator = tqdm.tqdm(
            list(range(hdf5_video_count)), desc="Fetching HDF5 video metadata"
        )
        for video_idx in iterator:
            parser = hdf5_parser.TAOVideoParser(
                hdf5_file_path=hdf5_file_path,
                video_idx=video_idx,
                convert_to_rgb=convert_to_rgb,
            )
            assert len(parser) > 0
            if skip_private_datasets:
                if parser.metadata["dataset"] in tao_utils.private_dataset_names:
                    continue
            if len(parser.annotations_info) == 0 and prefix in ["train", "valid"]:
                logger.warning(f"no annotation found for video: {parser.name}")
                continue  # we skip these videos since they cannot be used for anything
            video_parsers.append(parser)
        return video_parsers

    @staticmethod
    def _create_raw_video_parsers(
        data_root_path: typing.AnyStr,
        annot_file_path: typing.AnyStr,
        zip_file_path: typing.AnyStr,
        import_only_annot_frames: bool,
        skip_private_datasets: bool,
        prefix: typing.AnyStr,
        convert_to_rgb: bool,
    ) -> typing.Sequence[tao_utils.TAOVideoParserBase]:
        if import_only_annot_frames and prefix == "test":
            return []

        # first, open the JSON file that contains all the annotations for the specified split
        logger.info(f"\tLoading annotations from: {annot_file_path}")
        assert os.path.isfile(annot_file_path)
        with open(annot_file_path, "r") as fd:
            dataset_info = json.load(fd)
        logger.info("\tGenerating reverse lookup maps...")
        TAODataModule._add_reverse_lookup_maps_to_dataset_info(dataset_info)

        # next, open the zip & get ready to parse all the metadata from the JSON+zip simultaneously
        video_parsers = []
        logger.info(f"\tParsing content for zip file: {zip_file_path}")
        # note: zip will stay opened in memory until all parsers are destroyed!
        # (should not be big enough to cause issues though)
        zfd = zipfile.ZipFile(zip_file_path, "r")
        logger.info(f"\tCreating video parsers...")
        iterator = tqdm.tqdm(dataset_info["videos"], desc="Fetching video metadata")
        for video_info in iterator:  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            if skip_private_datasets:
                if video_info["metadata"]["dataset"] in tao_utils.private_dataset_names:
                    continue
            parser = raw_parser.TAOVideoParser(
                data_root_path=data_root_path,
                video_info=video_info,
                dataset_info=dataset_info,
                dataset_zip_fd=zfd,
                import_only_annot_frames=import_only_annot_frames,
                convert_to_rgb=convert_to_rgb,
            )
            assert len(parser) > 0
            if len(parser.annotations_info) == 0 and prefix in ["train", "valid"]:
                logger.warning(f"no annotation found for video: {video_info['name']}")
                continue  # we skip these videos since they cannot be used for anything
            video_parsers.append(parser)
        return video_parsers

    @staticmethod
    def _verify_split_overlap(
        parsers: typing.Iterable[typing.Sequence[tao_utils.TAOVideoParserBase]],
    ):
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
