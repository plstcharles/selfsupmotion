# this module will display the content of HDF5 files packing TAO video tracks
import logging

import cv2 as cv

import selfsupmotion.data.tao.data_module as tao_data_module

logger = logging.getLogger(__name__)
import argparse


def main(args):

    import_only_annot_frames = True

    data_module = tao_data_module.TAODataModule(
        data_root_path=args.data_root_path,
        use_hdf5_packages=True,
        import_only_annot_frames=import_only_annot_frames,
        skip_private_datasets=True,
        convert_to_rgb=False,
    )

    dataset_map = {
        "train": data_module.train_parsers,
        "valid": data_module.valid_parsers,
        "test": data_module.test_parsers,
    }

    for prefix, dataset in dataset_map.items():
        for video_parser in dataset:
            logger.info(f"displaying: {video_parser.name}")
            for frame_idx in range(len(video_parser)):
                frame_dict = video_parser[frame_idx]
                image = frame_dict["image_data"]
                got_gt = False
                for tid, track_info in frame_dict["tracks"].items():
                    import random

                    rnd = lambda: random.randint(0, 255)
                    random.seed(int(tid))
                    r, g, b = rnd(), rnd(), rnd()
                    # print("#%02X%02X%02X" % (r(), r(), r()))
                    if track_info["annotation"]:
                        bbox = track_info["annotation"]["bbox"]
                        cv.rectangle(
                            image,
                            pt1=(bbox[0], bbox[1]),
                            pt2=(bbox[0] + bbox[2], bbox[1] + bbox[3]),
                            color=(r, g, b),
                            thickness=1,
                        )
                        got_gt = True
                image = cv.resize(
                    image,
                    dsize=(-1, -1),
                    fx=args.fx,
                    fy=args.fy,
                    interpolation=cv.INTER_NEAREST,
                )
                cv.imshow("test", image)
                cv.waitKey(0 if got_gt else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Vis tool for TAO")
    parser.add_argument(
        "--data-root-path", type=str, default="/nfs/server/datasets/tao/"
    )
    parser.add_argument("--fx", type=int, default=4)
    parser.add_argument("--fy", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
    print("all done")

    # we want to highlight how much better representations are when using temporal-pairs instead of synthetic-aug-pairs

    # what % of the time is the model capable of finding the 'real' match among X candidates when the candidates come from Y frames in the future
    # draw this as a curve: X axis = for each pair (t, t+x) at different offsets of x, Y axis = average % of correct matches found across all seqs
