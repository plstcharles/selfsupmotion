import typing

import albumentations
import cv2 as cv
import numpy as np
import PIL
import torchvision
import random

try:
    import thelper
    thelper_available = True
except ImportError:
    thelper_available = False

import selfsupmotion.data.utils


class SimSiamFramePairDataTransform(object):
    """
    Transforms for SimSiam + Objectron:

        _generate_obj_crops(size=320)              (custom for Objectron specifically)
        RandomResizedCrop(size=self.input_height)  (grabs a fixed-size subregion to encode)
        RandomHorizontalFlip()                     (this and following ops apply to all frames)
        RandomApply([color_jitter], p=0.8)
        RandomGrayscale(p=0.2)
        GaussianBlur(kernel_size=int(0.1 * self.input_height))
        transforms.ToTensor()

    (note: the transform list is copied and adapted from the SimCLR transforms)
    """

    def _generate_obj_crops(self, sample: typing.Dict, crop_height: typing.Union[int, str]):
        """
        This operation will crop all frames in a sequence based on the object center location
        in the first frame. This will allow the model to perceive some of the camera movement.
        """
        assert isinstance(sample, dict) and "IMAGE" in sample and "CENTROID_2D_IM" in sample
        assert len(sample["IMAGE"].shape) == 4 and sample["IMAGE"].shape[1:] == (640, 480, 3)
        assert len(sample["CENTROID_2D_IM"].shape) == 2 and sample["CENTROID_2D_IM"].shape[-1] == 2
        assert self.crop_strategy in ["centroid", "bbox"]
        # get top-left/bottom-right coords for object of interest in first frame (0-th index)
        if self.crop_strategy == "centroid":
            if isinstance(crop_height, int):
                tl = (int(round(sample["CENTROID_2D_IM"][0, 0] - crop_height / 2)),
                      int(round(sample["CENTROID_2D_IM"][0, 1] - crop_height / 2)))
                br = (tl[0] + crop_height, tl[1] + crop_height)
            else:
                assert crop_height == "auto", "unexpected crop height arg"
                base_pts = np.asarray([pt for pt in sample["POINTS"][0]])
                real_tl = (base_pts[:, 0].min(), base_pts[:, 1].min())
                real_br = (base_pts[:, 0].max(), base_pts[:, 1].max())
                max_size = max(real_br[0] - real_tl[0], real_br[1] - real_tl[1]) * 1.1  # 10% extra
                tl = (int(round(sample["CENTROID_2D_IM"][0, 0] - max_size / 2)),
                      int(round(sample["CENTROID_2D_IM"][0, 1] - max_size / 2)))
                br = (int(round(tl[0] + max_size)), int(round(tl[1] + max_size)))
        elif self.crop_strategy=="bbox":
            tl = (int(sample["POINTS"][0,:,0].min()), int(sample["POINTS"][0,:,1].min()))
            br = (int(sample["POINTS"][0,:,0].max()), int(sample["POINTS"][0,:,1].max()))
        else:
            raise ValueError(f"Invalid cropping stragegy: {self.crop_strategy}")
        # get crops one at a time for all frames in the seq, for all seqs in the minibatch
        if tl == br:
            print(f"Annotation error on {sample['UID']}, moving on w/ hard-sized crop!")
            new_crop_height = 360
            tl = (int(round(sample["CENTROID_2D_IM"][0, 0] - new_crop_height / 2)),
                  int(round(sample["CENTROID_2D_IM"][0, 1] - new_crop_height / 2)))
            br = (tl[0] + new_crop_height, tl[1] + new_crop_height)
        output_crop_seq, output_keypoints, output_crop_offsets = [], [], []
        for frame_idx, (frame, kpts) in enumerate(zip(sample["IMAGE"], sample["POINTS"])):
            if thelper_available:
                crop = thelper.draw.safe_crop(image=frame, tl=tl, br=br)
            else:
                crop = selfsupmotion.data.utils.safe_crop(image=frame, tl=tl, br=br)
            assert crop is not None and crop.size > 0
            output_crop_seq.append(crop)
            offset_coords = (tl[0], tl[1], 0, 0)
            output_keypoints.append(np.subtract(sample["POINTS"][frame_idx], offset_coords))
            output_crop_offsets.append(offset_coords[:2])
        assert "OBJ_CROPS" not in sample
        sample["OBJ_CROPS"] = output_crop_seq
        sample["OBJ_CROPS_OFFSETS"] = output_crop_offsets
        sample["POINTS"] = output_keypoints
        return sample

    def __init__(
            self,
            crop_height: typing.Union[int, typing.AnyStr] = 320,
            input_height: int = 224,
            gaussian_blur: bool = True,
            jitter_strength: float = 1.,
            drop_orig_image: bool = True,
            crop_scale: typing.Tuple[float, float] = (0.2, 1.0),
            augmentation = True, #Will be used to disable augmentation on inference / validation.
            crop_strategy = "centroid",
            sync_hflip= False,
    ) -> None:
        self.crop_height = crop_height
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.drop_orig_image = drop_orig_image
        self.enable_augmentation = augmentation
        self.crop_strategy = crop_strategy
        self.sync_hflip = sync_hflip
        self.crop_scale = crop_scale

        bbox_transforms = [
            albumentations.LongestMaxSize(
                max_size=224
            ),
            albumentations.PadIfNeeded(
                min_height=224,
                min_width=224,
                border_mode=0,
            )
        ]
        assert self.crop_strategy in ["centroid","bbox"]

        if self.enable_augmentation:
            augment_transforms = [
                albumentations.RandomResizedCrop(
                    height=self.input_height,
                    width=self.input_height,
                    scale=self.crop_scale,
                    ratio=(1.0, 1.0),  # should not change aspect ratio to keep geometry intact..
                ),
            ]
            if self.crop_strategy == "bbox":
                augment_transforms = bbox_transforms + augment_transforms
            augment_transforms.extend([
                albumentations.ColorJitter(
                    brightness=0.4 * self.jitter_strength,
                    contrast=0.4 * self.jitter_strength,
                    saturation=0.4 * self.jitter_strength,
                    hue=0.1 * self.jitter_strength,
                    p=0.8,
                ),
                albumentations.ToGray(p=0.2),
            ])
            if self.gaussian_blur:
                # @@@@@ TODO: check what kernel size is best? is auto good enough?
                #kernel_size = int(0.1 * self.input_height)
                #if kernel_size % 2 == 0:
                #    kernel_size += 1
                augment_transforms.append(albumentations.GaussianBlur(
                    blur_limit=(3, 5),
                    #blur_limit=kernel_size,
                    #sigma_limit=???
                    p=0.5,
                ))
        else:
            augment_transforms = bbox_transforms
        self.augment_transform = albumentations.Compose(augment_transforms)

        self.convert_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.sync_hflip_transform = albumentations.Compose([
            albumentations.HorizontalFlip(p=1),  # @@@@@@@@@ BAD W/O SEED WRAPPER?
        ])

    def __call__(self, sample):
        assert isinstance(sample, dict)
        # first, add the object crops to the sample dict
        sample = self._generate_obj_crops(sample, self.crop_height)
        output_crops, output_keypoints, output_offsets, output_scale, output_hflipped = [], [], [], [], []
        flip = self.sync_hflip and random.choice([True, False])
        for crop_idx in range(len(sample["OBJ_CROPS"])):
            # we'll use two fake keypoints to keep track of the crop/scale transform for inversion
            assert sample["OBJ_CROPS"][crop_idx].shape[0] == sample["OBJ_CROPS"][crop_idx].shape[1]
            ref_tl_kpt = np.asarray((0, 0, 0, 0)).astype(np.float64)
            ref_br_kpt = np.asarray((*sample["OBJ_CROPS"][crop_idx].shape[:2], 0, 0)).astype(np.float64)
            aug_crop = self.augment_transform(
                image=sample["OBJ_CROPS"][crop_idx],
                keypoints=[*sample["POINTS"][crop_idx], ref_tl_kpt, ref_br_kpt],
                # the "xy" format somehow breaks when we have 2-coord kpts, this is why we pad to 4...
                keypoint_params=albumentations.KeypointParams(format="xysa", remove_invisible=False),
            )
            # extract the fake keypoints right away and compute the proper scale/offset based on them
            temp_aug_kpts = aug_crop["keypoints"]
            extra_crop_offset = (ref_tl_kpt - temp_aug_kpts[-2])[:2]
            new_crop_size = (np.asarray(temp_aug_kpts[-1]) - temp_aug_kpts[-2])[:2].astype(np.float32)
            assert len(np.unique(new_crop_size)) == 1  # if this blows up, we're using a bad geom op
            if flip:
                aug_crop = self.sync_hflip_transform(
                    image=aug_crop["image"],
                    keypoints=temp_aug_kpts,
                    keypoint_params=albumentations.KeypointParams(format="xysa", remove_invisible=False)
                )
            output_hflipped.append(flip)
            output_scale.append(new_crop_size[0] / sample["OBJ_CROPS"][crop_idx].shape[0])
            output_offsets.append(np.add(sample["OBJ_CROPS_OFFSETS"][crop_idx], extra_crop_offset))
            output_keypoints.append(aug_crop["keypoints"][:-2])  # minus the two fake keypoints
            output_crops.append(np.asarray(self.convert_transform(PIL.Image.fromarray(aug_crop["image"]))))
        sample["OBJ_CROPS"] = np.asarray(output_crops)
        sample["OBJ_CROPS_OFFSETS"] = np.asarray(output_offsets).astype(np.float32)
        sample["OBJ_CROPS_SCALE"] = np.asarray(output_scale).astype(np.float32)
        sample["OBJ_CROPS_HFLIPPED"] = np.asarray(output_hflipped).astype(np.int32)
        # finally, scrap the dumb padding around the 2d keypoints
        sample["POINTS"] = np.asarray([pts for pts in np.asarray(output_keypoints)[..., :2].astype(np.float32)])
        if self.drop_orig_image:
            del sample["IMAGE"]
            del sample["CENTROID_2D_IM"]
        return sample
