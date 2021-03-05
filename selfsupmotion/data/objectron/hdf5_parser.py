"""Objectron dataset parser module.

This module contains dataset parsers used to load the HDF5 archive of the Objectron dataset.
See https://github.com/google-research-datasets/Objectron for more info.
"""

import collections
import os
import typing
import time

import cv2 as cv
import h5py
import numpy as np
import pickle
import pytorch_lightning
import torch.utils.data
import tqdm

try:
    import turbojpeg
    turbojpeg = turbojpeg.TurboJPEG()
except ImportError:
    turbojpeg = None

import selfsupmotion.data.objectron.data_transforms
import selfsupmotion.data.objectron.sequence_parser
import selfsupmotion.data.objectron.utils
import selfsupmotion.data.utils


class ObjectronHDF5SequenceParser(torch.utils.data.Dataset):
    """Objectron HDF5 dataset parser.

    This class can be used to parse the (non-official) Objectron HDF5. More specifically, it allows
    random access over all sequences of the original dataset. See the HDF5 extractor's module for
    information on what the HDF5 files contain.
    """

    def __init__(
            self,
            hdf5_path: typing.AnyStr,
            seq_subset: typing.Optional[typing.Sequence[typing.AnyStr]] = None,  # default => use all
    ):
        self.hdf5_path = hdf5_path
        assert os.path.exists(self.hdf5_path), f"invalid dataset path: {self.hdf5_path}"
        all_objects = selfsupmotion.data.objectron.sequence_parser.ObjectronSequenceParser.all_objects
        if not seq_subset:
            self.objects = all_objects
        else:
            # assume we provided subset as a list of '<objname>/<seqid>' or '<objname>' strings
            objects = sorted([obj for obj in np.unique([s.split("/")[0] for s in seq_subset])])
            assert all([obj in all_objects for obj in objects]), "invalid object name used in filter"
            self.objects = objects
        self.seq_subset = seq_subset
        self.seq_name_map = {}
        with h5py.File(self.hdf5_path, mode="r") as fd:
            self.target_subsampl_rate = fd.attrs["target_subsampl_rate"]
            self.objectron_subset = fd.attrs["objectron_subset"]
            self.data_fields = fd.attrs["data_fields"]
            self.attr_fields = fd.attrs["attr_fields"]
            for object in self.objects:
                if object not in fd:
                    print(f"missing '{object}' group in dataset, skipping...")
                    continue
                for seq_id, seq_data in fd[object].items():
                    seq_name = object + "/" + seq_id
                    if not self.seq_subset or seq_name in self.seq_subset:
                        self.seq_name_map[len(self.seq_name_map)] = seq_name
        assert len(self.seq_name_map) > 0, "invalid sequence subset specified for given archive"
        if not self.seq_subset:
            self.seq_subset = [s for s in self.seq_name_map.values()]
        self.local_fd = None  # will be opened by worker on first getitem call

    def __len__(self):
        return len(self.seq_name_map)

    def __getitem__(self, idx):
        assert 0 <= idx < len(self.seq_name_map), "sequence index oob"
        if self.local_fd is None:
            self.local_fd = h5py.File(self.hdf5_path, mode="r")
        return self.local_fd[self.seq_name_map[idx]]

    @staticmethod
    def _is_frame_valid(dataset, frame_idx):
        # to keep things light and fast (enough), we'll only look at object centroids
        im_coords = dataset["CENTROID_2D_IM"][frame_idx]
        # in short, if any centroid is outside the image frame by more than its size, it's bad
        # (this will catch crazy-bad object coordinates that would lead to insanely big crops)
        return -640 < im_coords[0] < 1280 and -480 < im_coords[1] < 960

    @staticmethod
    def _decode_jpeg(data):
        if turbojpeg is not None:
            image = turbojpeg.decode(data)
        else:
            image = cv.imdecode(data, cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        return image


class ObjectronHDF5FrameTupleParser(ObjectronHDF5SequenceParser):
    """Objectron HDF5 dataset parser.

    This class can be used to parse the (non-official) Objectron HDF5. More specifically, it allows
    random access over frame tuples of the original dataset. See the HDF5 extractor's module for
    information on what the HDF5 files contain.

    What's a tuple in this context, you ask? Well, it's a series of consecutive frames (which
    might already be subsampled in the HDF5 writer) that are also separated by `frame_offset`
    extra frames. Consecutive tuples are then also separated by `tuple_offset` frames.
    """

    # @@@@ TODO in the future: create pairs on smallest pt reproj errors?

    def __init__(
            self,
            hdf5_path: typing.AnyStr,
            seq_subset: typing.Optional[typing.Sequence[typing.AnyStr]] = None,  # default => use all
            tuple_length: int = 2,  # by default, we will create pairs of frames
            frame_offset: int = 1,  # by default, we will create tuples from consecutive frames
            tuple_offset: int = 2,  # by default, we will skip a frame between tuples
            keep_only_frames_with_valid_kpts: bool = False,  # set to true when training on kpts
            keep_only_frames_with_valid_ground: bool = False,  # set to true when training w/ ground plane
            _target_fields: typing.Optional[typing.List[typing.AnyStr]] = None,
            _transforms: typing.Optional[typing.Any] = None,
            _resort_keypoints: bool = False,
    ):
        # also make a local var for the archive name, as we don't want to overwrite cache for each
        hdf5_name = os.path.basename(hdf5_path)  # yes, it's normal that this looks "unused" (it's used)
        cache_hash = selfsupmotion.data.utils.get_params_hash(
            {k: v for k, v in vars().items() if not k.startswith("_") and k != "self"})
        super().__init__(hdf5_path=hdf5_path, seq_subset=seq_subset)
        assert tuple_length >= 1 and frame_offset >= 1 and tuple_offset >= 1, "invalid tuple params"
        self.tuple_length = tuple_length
        self.frame_offset = frame_offset
        self.tuple_offset = tuple_offset
        self.keep_only_frames_with_valid_kpts = keep_only_frames_with_valid_kpts
        self.keep_only_frames_with_valid_ground = keep_only_frames_with_valid_ground
        self.target_fields = self.data_fields if not _target_fields else _target_fields
        self.transforms = _transforms
        cache_path = os.path.join(os.path.dirname(hdf5_path), cache_hash + ".pkl")
        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as fd:
                self.frame_pair_map = pickle.load(fd)
        else:
            self.frame_pair_map = self._fetch_tuple_metadata()
            with open(cache_path, "wb") as fd:
                pickle.dump(self.frame_pair_map, fd)
        self.resort_keypoints = _resort_keypoints

    def _fetch_tuple_metadata(self):
        tuple_map = {}
        assert len(self.seq_subset) > 0
        with h5py.File(self.hdf5_path, mode="r") as fd:
            progress_bar = tqdm.tqdm(
                self.seq_subset, total=len(self.seq_subset),
                desc=f"pre-fetching tuple metadata"
            )
            for seq_name in progress_bar:
                seq_data = fd[seq_name]
                seq_len = len(seq_data["IMAGE"])
                if seq_len <= self.frame_offset:  # skip, can't create frame tuples
                    continue
                seq_image_size = (seq_data.attrs["IMAGE_WIDTH"], seq_data.attrs["IMAGE_HEIGHT"])
                # note: the frame idx here is not the real index if the sequence was subsampled
                tuple_start_idx = 0
                while tuple_start_idx < seq_len:
                    candidate_tuple_vals = collections.defaultdict(list)
                    for tuple_idx_offset in range(self.tuple_length):
                        curr_frame_idx = tuple_start_idx + tuple_idx_offset * self.frame_offset
                        if curr_frame_idx >= seq_len:
                            break  # if we're oob for any element in the tuple, break it off
                        if not self._is_frame_valid(seq_data, curr_frame_idx):
                            break  # if an element has a bad frame, break it off
                        # we also get the 2d point projections here
                        curr_pts_2d = seq_data["POINT_2D"][curr_frame_idx].reshape((-1, 3))[:, :2]
                        assert len(curr_pts_2d) == 9, "did we not melt the dataset down to 1 instance?"
                        if self.keep_only_frames_with_valid_kpts:
                            # arbitrary pick: min valid keypoint count = 3
                            good_kpts = np.logical_and(curr_pts_2d <= 1, curr_pts_2d >= 0).all(axis=1)
                            if sum(good_kpts) < 3:
                                break
                        if self.keep_only_frames_with_valid_ground:
                            # check to make sure the bounding box has 4 points sitting on the ground plane
                            curr_pts_3d = seq_data["POINT_3D"][curr_frame_idx].reshape((-1, 3))[1:]
                            plane_normal = seq_data["PLANE_NORMAL"][curr_frame_idx]
                            plane_center = seq_data["PLANE_CENTER"][curr_frame_idx]
                            plane_d = np.dot(-plane_normal, plane_center)
                            kpts_plane_dists = np.asarray([
                                selfsupmotion.data.utils.distance_between_point_and_plane(
                                    pt[0], pt[1], pt[2],
                                    plane_normal[0], plane_normal[1], plane_normal[2], plane_d,
                                ) for pt in curr_pts_3d
                            ])
                            kpts_touching_ground = sum([np.isclose(dist, 0., atol=0.005) for dist in kpts_plane_dists])
                            if kpts_touching_ground != 4:
                                break
                        candidate_tuple_vals["POINT_2D"].append(np.multiply(curr_pts_2d, seq_image_size))
                        candidate_tuple_vals["frame_idxs"].append(curr_frame_idx)
                    if len(candidate_tuple_vals["frame_idxs"]) == self.tuple_length:  # if it's all good, keep it
                        candidate_tuple_vals["seq_name"] = seq_name
                        tuple_map[len(tuple_map)] = dict(candidate_tuple_vals)
                    tuple_start_idx += self.tuple_offset
        return tuple_map

    def __len__(self):
        return len(self.frame_pair_map)

    def __getitem__(self, idx):
        assert 0 <= idx < len(self.frame_pair_map), "frame pair index oob"
        if self.local_fd is None:
            self.local_fd = h5py.File(self.hdf5_path, mode="r")
        meta = self.frame_pair_map[idx]
        seq_name = meta["seq_name"]
        seq_data = self.local_fd[seq_name]
        assert seq_data.attrs["IMAGE_HEIGHT"] == 640 and seq_data.attrs["IMAGE_WIDTH"] == 480
        assert seq_data.attrs["ORIENTATION"] in ["portrait", "landscape"]
        sample = {
            field: np.stack([
                self._decode_jpeg(seq_data[field][frame_idx])
                if field == "IMAGE" else seq_data[field][frame_idx]
                for frame_idx in meta["frame_idxs"]
            ]) for field in self.target_fields
        }
        uid = f"hdf5_{seq_name}_" + str(seq_data["IMAGE_ID"][meta["frame_idxs"][0]])
        sample = {
            "SEQ_IDX": seq_name,
            "UID": uid, #TODO: Harmonize Unique Identifiers for evaluation.
            "FRAME_REAL_IDXS": np.asarray([seq_data["IMAGE_ID"][idx] for idx in meta["frame_idxs"]]),
            "FRAME_HDF5_IDXS": np.asarray(meta["frame_idxs"]),
            "POINTS": np.pad(np.asarray(meta["POINT_2D"]), ((0, 0), (0, 0), (0, 2))),  # pad 2d kpts for augments
            "CAT_ID": self.objects.index(seq_name.split("/")[0]),
            "IS_PORTRAIT": seq_data.attrs["ORIENTATION"] == "portrait",
            "IMAGE_WIDTH": seq_data.attrs["IMAGE_WIDTH"],
            "IMAGE_HEIGHT": seq_data.attrs["IMAGE_HEIGHT"],
            **sample,
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        # resort after transforms so flips can be used
        if self.resort_keypoints:
            sample["sorting_good"] = selfsupmotion.data.objectron.utils.sort_sample_keypoints(sample)
        return sample


class ObjectronFramePairDataModule(pytorch_lightning.LightningDataModule):

    name = "objectron_frame_pair"

    def __init__(
            self,
            hdf5_path: typing.AnyStr,
            tuple_length: int = 2,
            frame_offset: int = 1,
            tuple_offset: int = 2,
            keep_only_frames_with_valid_kpts: bool = False,
            keep_only_frames_with_valid_ground: bool = False,
            input_height: int = 224,
            gaussian_blur: bool = True,
            jitter_strength: float = 1.0,
            valid_split_ratio: float = 0.1,
            num_workers: int = 8,
            batch_size: int = 256,
            split_seed: int = 1337,
            pin_memory: bool = True,
            drop_last: bool = False,
            crop_height: typing.Union[int, typing.AnyStr] = "auto",
            crop_scale: tuple = (0.2, 1),
            crop_strategy: str = "centroid",
            sync_hflip = False,
            resort_keypoints: bool = False,
            drop_orig_image: bool = True,
            *args: typing.Any,
            **kwargs: typing.Any,
    ):
        super().__init__(*args, **kwargs)
        self.image_size = input_height
        self.dims = (3, input_height, input_height)
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.valid_split_ratio = valid_split_ratio
        self.hdf5_path = hdf5_path
        self.tuple_length = tuple_length
        self.frame_offset = frame_offset
        self.tuple_offset = tuple_offset
        self.keep_only_frames_with_valid_kpts = keep_only_frames_with_valid_kpts
        self.keep_only_frames_with_valid_ground = keep_only_frames_with_valid_ground
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.split_seed = split_seed if split_seed is not None else 1337  # keep this static between runs
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.crop_height = crop_height
        self.crop_scale = crop_scale
        self.crop_strategy = crop_strategy
        self.sync_hflip = sync_hflip
        self.resort_keypoints = resort_keypoints
        self.drop_orig_image = drop_orig_image
        # create temp dataset to get total sequence/sample count
        dataset = ObjectronHDF5FrameTupleParser(
            hdf5_path=self.hdf5_path,
            seq_subset=None,  # grab all sequences in first pass, then use em for split
            tuple_length=self.tuple_length,
            frame_offset=self.frame_offset,
            tuple_offset=self.tuple_offset,
            keep_only_frames_with_valid_kpts=self.keep_only_frames_with_valid_kpts,
            keep_only_frames_with_valid_ground=self.keep_only_frames_with_valid_ground,
        )
        # we do the split in a 2nd pass and re-parse the subsets
        # ...caching will be 2x longer, but it's the easiest solution, and it costs ~5 minutes once
        pre_seed_state = np.random.get_state()
        np.random.seed(self.split_seed)
        seq_subset_idxs = np.random.permutation(len(dataset.seq_subset))
        np.random.set_state(pre_seed_state)
        self.valid_seq_count = int(len(seq_subset_idxs) * self.valid_split_ratio)
        self.train_seq_count = len(seq_subset_idxs) - self.valid_seq_count
        self.valid_seq_subset = sorted([dataset.seq_subset[idx] for idx in seq_subset_idxs[:self.valid_seq_count]])
        self.train_seq_subset = sorted([dataset.seq_subset[idx] for idx in seq_subset_idxs[self.valid_seq_count:]])
        self.valid_sample_subset = [idx for idx, m in dataset.frame_pair_map.items() if m["seq_name"] in self.valid_seq_subset]
        self.train_sample_subset = [idx for idx, m in dataset.frame_pair_map.items() if m["seq_name"] in self.train_seq_subset]
        self.valid_sample_count = len(self.valid_sample_subset)
        self.train_sample_count = len(self.train_sample_subset)
        assert len(np.intersect1d(self.valid_seq_subset, self.train_seq_subset)) == 0
        assert len(np.intersect1d(self.valid_sample_count, self.train_sample_count)) == 0
        assert self.train_sample_count > 0 and self.valid_sample_count > 0
        self.train_dataset = None  # will be initialized once setup is called
        self.val_dataset = None

    def setup(self, stage=None):
        if self.val_transforms is None:
            self.val_transforms = self.val_transform()
        if self.train_transforms is None:
            self.train_transforms = self.train_transform()
        target_fields = [
            "IMAGE", "CENTROID_2D_IM", "POINT_3D",
            "PROJECTION_MATRIX", "VIEW_MATRIX",
            "PLANE_CENTER", "PLANE_NORMAL",
            "EXTRINSIC_MATRIX", "INTRINSIC_MATRIX",
        ]
        self.val_dataset = ObjectronHDF5FrameTupleParser(
            hdf5_path=self.hdf5_path,
            seq_subset=self.valid_seq_subset,
            tuple_length=self.tuple_length,
            frame_offset=self.frame_offset,
            tuple_offset=self.tuple_offset,
            keep_only_frames_with_valid_kpts=self.keep_only_frames_with_valid_kpts,
            keep_only_frames_with_valid_ground=self.keep_only_frames_with_valid_ground,
            _target_fields=target_fields,
            _transforms=self.val_transforms,
            _resort_keypoints=self.resort_keypoints,
        )
        self.train_dataset = ObjectronHDF5FrameTupleParser(
            hdf5_path=self.hdf5_path,
            seq_subset=self.train_seq_subset,
            tuple_length=self.tuple_length,
            frame_offset=self.frame_offset,
            tuple_offset=self.tuple_offset,
            keep_only_frames_with_valid_kpts=self.keep_only_frames_with_valid_kpts,
            keep_only_frames_with_valid_ground=self.keep_only_frames_with_valid_ground,
            _target_fields=target_fields,
            _transforms=self.train_transforms,
            _resort_keypoints=self.resort_keypoints,
        )

    def train_dataloader(self, evaluation=False) -> torch.utils.data.DataLoader:
        if evaluation:  # We don't want transformations for finding nearest neighbors.
            self.train_dataset.transforms = self.val_transforms
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self, evaluation=False) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=selfsupmotion.data.utils.ConstantRandomOrderSampler(self.val_dataset),
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )

    def train_transform(self):
        return selfsupmotion.data.objectron.data_transforms.SimSiamFramePairDataTransform(
            crop_height=self.crop_height,
            input_height=self.image_size,
            gaussian_blur=self.gaussian_blur,
            jitter_strength=self.jitter_strength,
            drop_orig_image=self.drop_orig_image,
            crop_scale=self.crop_scale,
            augmentation=True,
            crop_strategy=self.crop_strategy,
            sync_hflip=self.sync_hflip
        )

    def val_transform(self):
        return selfsupmotion.data.objectron.data_transforms.SimSiamFramePairDataTransform(
            crop_height=self.crop_height,
            input_height=self.image_size,
            gaussian_blur=self.gaussian_blur,
            jitter_strength=self.jitter_strength,
            drop_orig_image=self.drop_orig_image,
            crop_scale=self.crop_scale,
            augmentation=False,
            crop_strategy=self.crop_strategy,
            sync_hflip=self.sync_hflip,
        )


if __name__ == "__main__":
    data_path = "/wdata/datasets/objectron/"
    hdf5_path = data_path + "extract_s5_raw.hdf5"
    config_path = "/home/perf6/dev/selfsupmotion/examples/local/config-kpts.yaml"
    display_keypoints = True
    batch_size = 1
    max_iters = 100000

    import yaml
    with open(config_path, "r") as stream:
        hyper_params = yaml.load(stream, Loader=yaml.FullLoader)

    import selfsupmotion.utils.reproducibility_utils
    hyper_params["seed"] = 0
    selfsupmotion.utils.reproducibility_utils.set_seed(hyper_params["seed"])

    dm = ObjectronFramePairDataModule(
        hdf5_path=hdf5_path,
        tuple_length=hyper_params.get("tuple_length"),
        frame_offset=hyper_params.get("frame_offset"),
        tuple_offset=hyper_params.get("tuple_offset"),
        keep_only_frames_with_valid_kpts=hyper_params.get("keep_only_frames_with_valid_kpts"),
        keep_only_frames_with_valid_ground=hyper_params.get("keep_only_frames_with_valid_ground"),
        input_height=hyper_params.get("input_height"),
        gaussian_blur=hyper_params.get("gaussian_blur"),
        jitter_strength=hyper_params.get("jitter_strength"),
        batch_size=batch_size,
        num_workers=0,
        drop_orig_image=False,
        crop_height=hyper_params.get("crop_height"),
        crop_scale=(hyper_params.get("crop_scale_min"), hyper_params.get("crop_scale_max")),
        crop_strategy=hyper_params.get("crop_strategy"),
        sync_hflip=hyper_params.get("sync_hflip"),
        resort_keypoints=hyper_params.get("resort_keypoints"),
    )
    assert dm.train_sample_count > 0
    dm.setup()
    loader = dm.train_dataloader()
    norm_std = np.asarray([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    norm_mean = np.asarray([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    display = batch_size == 1

    iter = 0
    init_time = time.time()
    for batch_idx, batch in enumerate(tqdm.tqdm(loader, total=len(loader))):
        if display:
            # if we get here, batch size is one, but we still have to unpack the tuple below
            sample_idx = 0
            display_frames, display_crops = [], []
            for frame_idx in range(len(batch["OBJ_CROPS"])):
                if not batch["IS_PORTRAIT"][sample_idx].item():
                    print("orly")
                pts = batch["POINTS"][sample_idx][frame_idx].reshape((9, 2)).numpy()
                pts_3d = batch["POINT_3D"][sample_idx][frame_idx].reshape((9, 3)).numpy()
                cam_intrinsics = batch["INTRINSIC_MATRIX"][sample_idx][frame_idx].reshape((3, 3)).numpy()
                # de-normalize and ready image for opencv display to show the result of transforms
                crop = batch["OBJ_CROPS"][sample_idx][frame_idx].squeeze(0).numpy().transpose((1, 2, 0))
                crop = (((crop * norm_std) + norm_mean) * 255).astype(np.uint8)
                frame = batch["IMAGE"][sample_idx][frame_idx].numpy()  # no need to denorm this one, it's the og
                frame = selfsupmotion.data.utils.draw_ground_plane(
                    frame,
                    batch["PLANE_CENTER"][sample_idx][frame_idx].numpy(),
                    batch["PLANE_NORMAL"][sample_idx][frame_idx].numpy(),
                    batch["PROJECTION_MATRIX"][sample_idx][frame_idx].numpy().reshape((4, 4)),
                )
                display_crops.append(selfsupmotion.data.utils.draw_2d_crop_kpts(crop, pts))
                display_frames.append(selfsupmotion.data.utils.draw_2d_frame_kpts(
                    frame,
                    pts,
                    batch["OBJ_CROPS_HFLIPPED"][sample_idx][frame_idx].item(),
                    batch["OBJ_CROPS_SCALE"][sample_idx][frame_idx].item(),
                    batch["OBJ_CROPS_OFFSETS"][sample_idx][frame_idx].numpy(),
                    crop.shape[1],
                ))

            cv.imshow("frames",
                cv.resize(
                    cv.hconcat(display_frames),
                    dsize=(-1, -1), fx=2, fy=2, interpolation=cv.INTER_NEAREST,
                )
            )
            cv.imshow("crops",
                cv.resize(
                    cv.hconcat(display_crops),
                    dsize=(-1, -1), fx=4, fy=4, interpolation=cv.INTER_NEAREST,
                )
            )
            cv.waitKey(0)

        iter += 1
        if max_iters is not None and iter > max_iters:
            break
    print(f"all done in {time.time() - init_time} seconds")
