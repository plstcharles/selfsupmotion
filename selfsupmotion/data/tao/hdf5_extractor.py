# this module will export TAO video tracks to HDF5 archives that can be randomly accessed
# (note that the export framerate is a parameter that must be controlled below)
import datetime
import json
import logging
import os
import typing
import shutil
import socket
import sys

import h5py
import numpy as np
import tqdm

import selfsupmotion.data.tao.data_module as tao_data_module
import selfsupmotion.data.tao.utils as tao_utils
import selfsupmotion.data.utils as data_utils
import selfsupmotion.utils.logging_utils as logging_utils

logger = logging.getLogger(__name__)


def _write_hdf5(
    hdf5_output_path: typing.AnyStr,
    dataset: typing.Sequence[tao_utils.TAOVideoParserBase],
    data_root_path: typing.AnyStr,
    export_only_annot_frames: bool,
    compression_type: typing.AnyStr,
):
    assert compression_type in tao_utils.supported_compression_types
    hdf5_tmp_path = hdf5_output_path + ".tmp"

    with h5py.File(hdf5_tmp_path, "w") as h5fd:
        h5fd.attrs["export_only_annot_frames"] = export_only_annot_frames
        h5fd.attrs["data_root_path"] = data_root_path
        h5fd.attrs["sha1"] = logging_utils.get_git_hash()
        h5fd.attrs["pyver"] = sys.version
        h5fd.attrs["hostname"] = socket.gethostname()
        h5fd.attrs["datetime"] = str(datetime.datetime.now())
        h5fd.attrs["compression_type"] = compression_type
        h5fd.attrs["video_count"] = len(dataset)

        metadata_out = h5fd.create_dataset(
            name="metadata",
            shape=(len(dataset),),
            dtype=h5py.special_dtype(vlen=np.uint8),
        )  # will contain pickled metadata only (i.e. everything that's loaded in the constructor)

        for video_idx, video_parser in enumerate(dataset):
            assert all([hasattr(video_parser, field_name) for field_name in tao_utils.video_attributes])
            metadata_to_export = {
                field_name: getattr(video_parser, field_name)
                for field_name in tao_utils.video_attributes
            }
            metadata_out[video_idx] = np.frombuffer(json.dumps(metadata_to_export).encode(), dtype=np.uint8)
            frame_count = len(video_parser)
            if compression_type in ["gzip", "lzf", "none", None]:
                dataset_kwargs = dict(
                    shape=(frame_count, video_parser.height, video_parser.width, 3),
                    chunks=(1, video_parser.height, video_parser.width, 3),
                    dtype=np.uint8,
                )
                if compression_type in ["gzip", "lzf"]:
                    dataset_kwargs["compression"] = compression_type
            else:
                dataset_kwargs = dict(
                    shape=(frame_count,),
                    dtype=h5py.special_dtype(vlen=np.uint8),
                )

            frame_data_out = h5fd.create_dataset(
                name=video_parser.name,
                **dataset_kwargs,
            )

            iterator = list(range(len(video_parser)))
            iterator = tqdm.tqdm(iterator, desc=f"{video_parser.name} ({video_idx + 1}/{len(dataset)})")
            curr_exported_frame_idx = 0
            for frame_idx in iterator:
                frame_data = video_parser[frame_idx]
                if compression_type in ["gzip", "lzf", "none", None]:
                    frame_data_out[curr_exported_frame_idx] = frame_data["image_data"]
                elif compression_type == "jpg":
                    image_data = data_utils.encode_jpeg(frame_data["image_data"])
                    frame_data_out[curr_exported_frame_idx] = np.frombuffer(image_data, dtype=np.uint8)
                else:
                    raise NotImplementedError
                curr_exported_frame_idx += 1
            assert frame_count == curr_exported_frame_idx

    # if we get here, we fully packaged the HDF5; overwrite any old ones w/ the new name
    shutil.move(hdf5_tmp_path, hdf5_output_path)


def main():

    data_root_path = "/nfs/server/datasets/tao/"
    export_only_annot_frames = True
    compression_type = "jpg"

    data_module = tao_data_module.TAODataModule(
        data_root_path=data_root_path,
        use_hdf5_packages=False,
        import_only_annot_frames=export_only_annot_frames,
        skip_private_datasets=True,
        convert_to_rgb=False,
    )

    dataset_map = {
        "train": data_module.train_parsers,
        "valid": data_module.valid_parsers,
        "test": data_module.test_parsers,
    }

    for prefix, dataset in dataset_map.items():
        hdf5_file_name = tao_utils.get_hdf5_file_name_from_prefix(prefix, export_only_annot_frames)
        hdf5_output_path = os.path.join(data_root_path, hdf5_file_name)
        _write_hdf5(
            hdf5_output_path=hdf5_output_path,
            dataset=dataset,
            data_root_path=data_root_path,
            export_only_annot_frames=export_only_annot_frames,
            compression_type=compression_type,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
    print("all done")
