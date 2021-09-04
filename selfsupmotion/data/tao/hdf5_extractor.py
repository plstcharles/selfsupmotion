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

supported_compression_types = ["jpg", "gzip", "lzf", "none", None]  # add lz4? add a video codec?


def _write_hdf5(
    hdf5_output_path: typing.AnyStr,
    dataset: typing.Sequence[tao_utils.TAOVideoParserBase],
    data_root_path: typing.AnyStr,
    export_only_annot_frames: bool,
    compression_type: typing.AnyStr,
):
    assert compression_type in supported_compression_types
    hdf5_tmp_path = hdf5_output_path + ".tmp"

    with h5py.File(hdf5_tmp_path, "w") as h5fd:
        h5fd.attrs["export_only_annot_frames"] = export_only_annot_frames
        h5fd.attrs["data_root_path"] = data_root_path
        h5fd.attrs["sha1"] = logging_utils.get_git_hash()
        h5fd.attrs["pyver"] = sys.version
        h5fd.attrs["hostname"] = socket.gethostname()
        h5fd.attrs["datetime"] = str(datetime.datetime.now())

        metadata_out = h5fd.create_dataset(
            name="metadata",
            shape=(len(dataset),),
            dtype=h5py.special_dtype(vlen=np.uint8),
        )  # will contain pickled metadata only (i.e. everything that's loaded in the constructor)

        for video_idx, video_parser in enumerate(dataset):
            assert all([hasattr(video_parser, field_name) for field_name in tao_utils.video_metadata_fields])
            metadata_to_export = {
                field_name: getattr(video_parser, field_name)
                for field_name in tao_utils.video_metadata_fields
            }
            metadata_out[video_idx] = np.frombuffer(json.dumps(metadata_to_export).encode(), dtype=np.uint8)

            if export_only_annot_frames:
                # we need to pre-count the number of frames with at least one non-None annotation in a track
                frame_tracks = [list(f["tracks"].values()) for f in video_parser.frames_metadata]
                frame_count = sum([any([t["annotation"] is not None for t in tracks]) for tracks in frame_tracks])
            else:
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
                if export_only_annot_frames:
                    if all([t["annotation"] is None for t in frame_data["tracks"].values()]):
                        continue
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
        skip_private_datasets=True,
    )

    dataset_map = {
        "train": data_module.train_video_parsers,
        "valid": data_module.valid_video_parsers,
        "test": data_module.test_video_parsers,
    }

    for prefix, dataset in dataset_map.items():
        if export_only_annot_frames and prefix == "test":
            continue
        only_annot_str = "_subsampl" if export_only_annot_frames else ""
        hdf5_output_path = os.path.join(data_root_path, f"tao_{prefix}{only_annot_str}.hdf5")
        _write_hdf5(
            hdf5_output_path=hdf5_output_path,
            dataset=dataset,
            data_root_path=data_root_path,
            export_only_annot_frames=export_only_annot_frames,
            compression_type=compression_type,
        )


if __name__ == "__main__":
    # we want to highlight how much better representations are when using temporal-pairs instead of synthetic-aug-pairs

    # what % of the time is the model capable of finding the 'real' match among X candidates when the candidates come from Y frames in the future
    # draw this as a curve: X axis = for each pair (t, t+x) at different offsets of x, Y axis = average % of correct matches found across all seqs

    logging.basicConfig(level=logging.INFO)
    main()
    print("all done")
