import json
import logging
import os
import typing

import h5py
import torch.utils.data

import selfsupmotion.data.utils as data_utils
import selfsupmotion.data.tao.utils as tao_utils

logger = logging.getLogger(__name__)


class TAOVideoParser(tao_utils.TAOVideoParserBase, torch.utils.data.Dataset):

    def __init__(
        self,
        hdf5_file_path: typing.AnyStr,  # path to the HDF5 that contains multiple
        video_idx: int,
        convert_to_rgb: bool,
    ):
        self.hdf5_file_path = hdf5_file_path
        self.hdf5_meta, video_meta = self._preparse_metadata(
            hdf5_file_path=self.hdf5_file_path,
            video_idx=video_idx,
        )
        super().__init__(
            id=video_meta["id"],
            name=video_meta["name"],
            metadata=video_meta["metadata"],
            width=video_meta["width"],
            height=video_meta["height"],
            neg_category_ids=video_meta["neg_category_ids"],
            not_exhaustive_category_ids=video_meta["not_exhaustive_category_ids"],
        )
        self.categories_info = video_meta["categories_info"]
        self.annotations_info = video_meta["annotations_info"]
        self.tracks_info = video_meta["tracks_info"]
        self.images_info = video_meta["images_info"]
        self.frames_metadata = video_meta["frames_metadata"]
        self.convert_to_rgb = convert_to_rgb
        self._internal_h5fd = None

    @staticmethod
    def _preparse_metadata(
        hdf5_file_path: typing.AnyStr,
        video_idx: int,
    ) -> typing.Tuple[typing.Dict, typing.Dict]:
        assert os.path.isfile(hdf5_file_path)
        with h5py.File(hdf5_file_path, "r") as h5fd:
            hdf5_meta = dict(h5fd.attrs)
            video_meta_buffer = h5fd["metadata"][video_idx].tobytes()
        video_meta = json.loads(video_meta_buffer.decode())
        assert all([n in tao_utils.video_attributes for n in video_meta])
        return hdf5_meta, video_meta

    def _load_frame_data(
        self,
        frame_idx: int,
        frame_dict: typing.Dict[typing.AnyStr, typing.Any],
    ):
        if self._internal_h5fd is None:
            self._internal_h5fd = h5py.File(self.hdf5_file_path, "r")
        compression_type = self.hdf5_meta["compression_type"]
        assert compression_type in tao_utils.supported_compression_types
        raw_data = self._internal_h5fd[self.name][frame_idx]
        if compression_type in ["gzip", "lzf", "none", None]:
            image_data = raw_data
        elif compression_type == "jpg":
            image_data = data_utils.decode_jpeg(
                data=raw_data,
                convert_to_rgb=self.convert_to_rgb,
            )
        else:
            raise AssertionError
        assert image_data.shape == (self.height, self.width, 3)
        frame_dict["image_data"] = image_data
