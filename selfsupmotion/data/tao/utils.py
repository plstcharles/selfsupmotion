import copy
import os
import typing


private_dataset_names = ["AVA", "HACS"]


def get_parent_zip_name_from_prefix(prefix: typing.AnyStr):
    """Returns the name of the zip file that contains the image data for the given split suffix."""
    if prefix.startswith("train"):
        return "1-TAO_TRAIN.zip"
    if prefix.startswith("valid"):
        return "2-TAO_VAL.zip"
    if prefix.startswith("test"):
        return "3-TAO_TEST.zip"
    raise NotImplementedError


def get_annot_file_name_from_prefix(prefix: typing.AnyStr):
    """Returns the name of the json file that contains the annotations for the given split suffix."""
    if prefix.startswith("train"):
        return "train_with_freeform.json"
    if prefix.startswith("valid"):
        return "validation_with_freeform.json"
    if prefix.startswith("test"):
        return "test_without_annotations.json"
    raise NotImplementedError


def check_if_images_are_on_disk(
    data_root_path: typing.AnyStr,
    rel_dir_path: typing.AnyStr,
    file_names: typing.Iterable[typing.AnyStr],
):
    """Checks whether the images (frames) of the sequence at the given path are available or not."""
    frames_root_path = os.path.join(data_root_path, "frames")
    frames_dir_path = os.path.join(frames_root_path, rel_dir_path)
    if not os.path.isdir(frames_dir_path):
        return False
    image_paths = [os.path.join(frames_root_path, file_name) for file_name in file_names]
    return len(image_paths) > 0 and all([os.path.isfile(p) for p in image_paths])


video_metadata_fields = [
    # from the dataset itself
    "id", "name", "metadata", "width", "height", "neg_category_ids", "not_exhaustive_category_ids",
    # defined internally, for debugging/speedup
    "categories_info", "annotations_info", "tracks_info", "images_info",
    # and the main lovebite
    "frames_metadata",
]


class TAOVideoParserBase:
    """Base class interface that shares its attributes across both raw and HDF5 parsers."""

    def __init__(
        self,
        id: int,  # unique identifier (integer) for this video
        name: typing.AnyStr,  # the 'name' is actually the relative path to the video folder
        metadata: typing.Dict,  # this contains mostly useless info (dataset name, user name)
        width: int,  # should be constant across the entire video sequence
        height: int,  # should be constant across the entire video sequence
        neg_category_ids: typing.Sequence[int],  # ids of categories IGNORED in this sequence
        not_exhaustive_category_ids: typing.Sequence[int],  # ids of annotated categories in this sequence
        # note: the 'non exhaustive' list means that not all tracked objects might have a known category
    ):
        # first, store the high-level info about the sequence itself
        self.id = id
        self.name = name
        self.metadata = metadata
        self.width = width
        self.height = height
        self.neg_category_ids = neg_category_ids
        self.not_exhaustive_category_ids = not_exhaustive_category_ids

        # the following reverse-info maps should be filled by the derived class itself
        self.categories_info = {}
        self.annotations_info = {}
        self.tracks_info = {}
        self.images_info = {}

        # the following map is the object that will be iterated over, providing info on each frame
        self.frames_metadata = []

    def __len__(self) -> int:
        """Returns the number of frames that can be loaded in the video sequence."""
        # note: not all frames will be annotated! (TAO is a roughly 1-annotated-FPS dataset)
        return len(self.frames_metadata)

    def _load_frame_data(self, frame_data: typing.Dict[typing.AnyStr, typing.Any]):
        """Loads all data that is not already in the 'metadata' package for a particular frame."""
        raise NotImplementedError  # must be implemented in the derived class

    def __getitem__(self, frame_idx: int) -> typing.Dict[typing.AnyStr, typing.Any]:
        """Returns a dictionary containing metadata (annotations) and image data for a video frame."""
        if frame_idx < 0:
            frame_idx = len(self) - frame_idx
        assert 0 <= frame_idx < len(self)
        # the metadata for the frame is all ready, we just need to load the image data itself
        frame_data = copy.deepcopy(self.frames_metadata[frame_idx])
        self._load_frame_data(frame_data)
        return frame_data
