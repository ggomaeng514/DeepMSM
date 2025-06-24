import warnings
from collections.abc import Callable, Sequence
from typing import Union, Tuple, Dict, Hashable, Mapping
from numpy import unravel_index, ceil
import numpy as np
import torch
from monai.transforms import Crop, CenterSpatialCrop, Cropd, MapTransform, LazyTrait
from monai.transforms.transform import Randomizable
from monai.transforms.utils import map_binary_to_indices
from monai.data import MetaTensor
from monai.config import KeysCollection, NdarrayOrTensor
from monai.utils import fall_back_tuple, ensure_tuple_rep


class LastTransfromMRIs(MapTransform):
    def __init__(
        self,
        keys = ("t1", "t1ce", "t2", "flair", "mask"),
        gaussian_noise: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.gaussian_noise = gaussian_noise

    def __call__(self, data):
        d = dict(data)
        
        t1 = d['t1']
        t1ce = d['t1ce']
        t2 = d['t2']
        flair = d['flair']
        mask = d['mask']
        
        t1_signal = t1[mask > 0]
        t1ce_signal = t1ce[mask > 0]
        t2_signal = t2[mask > 0]
        flair_signal = flair[mask > 0]
        meanv_t1, stdv_t1 = t1_signal.mean(), t1_signal.std()
        meanv_t1ce, stdv_t1ce = t1ce_signal.mean(), t1ce_signal.std()
        meanv_t2, stdv_t2 = t2_signal.mean(), t2_signal.std()
        meanv_flair, stdv_flair = flair_signal.mean(), flair_signal.std()
        
        t1 = (t1 - meanv_t1) / (stdv_t1 + 1e-8)
        t1ce = (t1ce - meanv_t1ce) / (stdv_t1ce + 1e-8)
        t2 = (t2 - meanv_t2) / (stdv_t2 + 1e-8)
        flair = (flair - meanv_flair) / (stdv_flair + 1e-8)
        
        mris = torch.cat([t1, t1ce, t2, flair], dim=0)
        if self.gaussian_noise:
            prob = random.random()
            if prob < 0.9:
                mris = mris + torch.randn_like(mris) * 0.05
                
        mris = mris * mask
        mris = mris.clamp(-2.5, 2.5)
        
        d['transformed_image'] = mris
        
        return d


class RandSpatialCropbyPos(Randomizable, Crop):
    """
    Crop image with random size or specific size ROI. It can crop at a random position as center
    or at the image center. And allows to set the minimum and maximum size to limit the randomly generated ROI.

    Note: even `random_size=False`, if a dimension of the expected ROI size is larger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than the expected ROI, and the cropped results
    of several images may not have exactly the same shape.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        roi_size: if `random_size` is True, it specifies the minimum crop region.
            if `random_size` is False, it specifies the expected ROI size to crop. e.g. [224, 224, 128]
            if a dimension of ROI size is larger than image size, will not crop that dimension of the image.
            If its components have non-positive values, the corresponding size of input image will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `roi_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        max_roi_size: if `random_size` is True and `roi_size` specifies the min crop region size, `max_roi_size`
            can specify the max crop region size. if None, defaults to the input image size.
            if its components have non-positive values, the corresponding size of input image will be used.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specific size ROI.
            if True, the actual size is sampled from `randint(roi_size, max_roi_size + 1)`.
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
    """

    def __init__(
        self,
        roi_size: Union[Sequence[int], int],
        max_roi_size: Union[Sequence[int], int, None] = None,
        random_center: bool = True,
        random_size: bool = False,
        allow_smaller: bool = False,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self.roi_size = roi_size
        self.max_roi_size = max_roi_size
        self.random_center = random_center
        self.random_size = random_size
        self.allow_smaller = allow_smaller
        self._size: Union[Sequence[int], None] = None
        self._slices: tuple[slice, ...]
        
    def randomize(self, img_size: Sequence[int], fg_indices: NdarrayOrTensor) -> None:
        self._size = fall_back_tuple(self.roi_size, img_size)
        if self.random_size:
            max_size = img_size if self.max_roi_size is None else fall_back_tuple(self.max_roi_size, img_size)
            if any(i > j for i, j in zip(self._size, max_size)):
                raise ValueError(f"min ROI size: {self._size} is larger than max ROI size: {max_size}.")
            self._size = tuple(self.R.randint(low=self._size[i], high=max_size[i] + 1) for i in range(len(img_size)))
            
        fg_indices = np.asarray(fg_indices) if isinstance(fg_indices, Sequence) else fg_indices
        if len(fg_indices) == 0:
            pos_ratio = 0 if len(fg_indices) == 0 else 1
            warnings.warn(
                f"Num foregrounds {len(fg_indices)}, "
                f"unable to generate class balanced samples, setting `pos_ratio` to {pos_ratio}."
            )
        # rand_state = np.random.random.__self__  # type: ignore
        random_int = self.R.randint(len(fg_indices))
        idx = fg_indices[random_int]
        center = unravel_index(idx, img_size)
        valid_center = [max(c_s // 2, min(c, s - c_s // 2)) for c, c_s, s in zip(center, self._size, img_size)]
        self._slices = tuple(slice(c - s // 2, c + (s + 1) // 2) for c, s in zip(valid_center, self._size))
        
    def __call__(self, img: torch.Tensor, randomize: bool = True, lazy: Union[bool, None] = None) -> torch.Tensor:  # type: ignore
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.

        """
        img_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
        if randomize:
            self.randomize(img_size)
        if self._size is None:
            raise RuntimeError("self._size not specified.")
        lazy_ = self.lazy if lazy is None else lazy
        if self.random_center:
            return super().__call__(img=img, slices=self._slices, lazy=lazy_)
        cropper = CenterSpatialCrop(self._size, lazy=lazy_)
        return super().__call__(img=img, slices=cropper.compute_slices(img_size), lazy=lazy_)
    

class RandScaleCropbyPos(RandSpatialCropbyPos):
    """
    Subclass of :py:class:`monai.transforms.RandSpatialCrop`. Crop image with
    random size or specific size ROI.  It can crop at a random position as
    center or at the image center.  And allows to set the minimum and maximum
    scale of image size to limit the randomly generated ROI.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        roi_scale: if `random_size` is True, it specifies the minimum crop size: `roi_scale * image spatial size`.
            if `random_size` is False, it specifies the expected scale of image size to crop. e.g. [0.3, 0.4, 0.5].
            If its components have non-positive values, will use `1.0` instead, which means the input image size.
        max_roi_scale: if `random_size` is True and `roi_scale` specifies the min crop region size, `max_roi_scale`
            can specify the max crop region size: `max_roi_scale * image spatial size`.
            if None, defaults to the input image size. if its components have non-positive values,
            will use `1.0` instead, which means the input image size.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specified size ROI by `roi_scale * image spatial size`.
            if True, the actual size is sampled from
            `randint(roi_scale * image spatial size, max_roi_scale * image spatial size + 1)`.
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
    """

    def __init__(
        self,
        roi_scale: Union[Sequence[float], float],
        max_roi_scale: Union[Sequence[float], float, None] = None,
        random_center: bool = True,
        random_size: bool = False,
        image_threshold: float = 0.0,
        allow_smaller: bool = False,
        lazy: bool = False,
    ) -> None:
        super().__init__(
            roi_size=-1, max_roi_size=None, random_center=random_center, random_size=random_size, allow_smaller=allow_smaller, lazy=lazy
        )
        self.roi_scale = roi_scale
        self.max_roi_scale = max_roi_scale
        self.image_threshold = image_threshold

    def get_max_roi_size(self, img_size):
        ndim = len(img_size)
        self.roi_size = [ceil(r * s) for r, s in zip(ensure_tuple_rep(self.roi_scale, ndim), img_size)]
        if self.max_roi_scale is not None:
            self.max_roi_size = [ceil(r * s) for r, s in zip(ensure_tuple_rep(self.max_roi_scale, ndim), img_size)]
        else:
            self.max_roi_size = None

    def randomize(self, image, label) -> None:
        img_size = image.peek_pending_shape() if isinstance(image, MetaTensor) else image.shape[1:]
        self.get_max_roi_size(img_size)
        fg_indices, bg_indices = map_binary_to_indices(label, image, self.image_threshold)
        super().randomize(img_size, fg_indices)

    def __call__(self, img: torch.Tensor, randomize: bool = True, lazy: Union[bool, None] = None) -> torch.Tensor:  # type: ignore
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.

        """
        self.get_max_roi_size(img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:])
        lazy_ = self.lazy if lazy is None else lazy
        return super().__call__(img=img, randomize=randomize, lazy=lazy_)
       
       
class RandCropbyPosd(Cropd, Randomizable):
    """
    Base class for random crop transform.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        cropper: random crop transform for the input image.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
    """

    backend = Crop.backend

    def __init__(self, keys: KeysCollection, cropper: Crop, allow_missing_keys: bool = False, lazy: bool = False):
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys, lazy=lazy)

    def set_random_state(self, seed: Union[int, None] = None, state: Union[np.random.RandomState, None] = None):
        super().set_random_state(seed, state)
        if isinstance(self.cropper, Randomizable):
            self.cropper.set_random_state(seed, state)
        return self

    def randomize(self, image, label) -> None:
        if isinstance(self.cropper, Randomizable):
            self.cropper.randomize(image, label)

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: Union[bool, None] = None) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        # the first key must exist to execute random operations
        first_item = d[self.first_key(d)]
        # label = d.get(self.label_key)
        # fg_indices, bg_indices = map_binary_to_indices(d.get(self.label_key), first_item, self.image_threshold)
        self.randomize(first_item, d.get(self.label_key))
        lazy_ = self.lazy if lazy is None else lazy
        if lazy_ is True and not isinstance(self.cropper, LazyTrait):
            raise ValueError(
                "'self.cropper' must inherit LazyTrait if lazy is True "
                f"'self.cropper' is of type({type(self.cropper)}"
            )
        for key in self.key_iterator(d):
            kwargs = {"randomize": False} if isinstance(self.cropper, Randomizable) else {}
            if isinstance(self.cropper, LazyTrait):
                kwargs["lazy"] = lazy_
            d[key] = self.cropper(d[key], **kwargs)  # type: ignore
        return d 
        
        
class RandScaleCropbyPosd(RandCropbyPosd):
    """
    Dictionary-based version :py:class:`monai.transforms.RandScaleCrop`.
    Crop image with random size or specific size ROI.
    It can crop at a random position as center or at the image center.
    And allows to set the minimum and maximum scale of image size to limit the randomly generated ROI.
    Suppose all the expected fields specified by `keys` have same shape.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        roi_scale: if `random_size` is True, it specifies the minimum crop size: `roi_scale * image spatial size`.
            if `random_size` is False, it specifies the expected scale of image size to crop. e.g. [0.3, 0.4, 0.5].
            If its components have non-positive values, will use `1.0` instead, which means the input image size.
        max_roi_scale: if `random_size` is True and `roi_scale` specifies the min crop region size, `max_roi_scale`
            can specify the max crop region size: `max_roi_scale * image spatial size`.
            if None, defaults to the input image size. if its components have non-positive values,
            will use `1.0` instead, which means the input image size.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specified size ROI by `roi_scale * image spatial size`.
            if True, the actual size is sampled from:
            `randint(roi_scale * image spatial size, max_roi_scale * image spatial size + 1)`.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
    """

    def __init__(
        self,
        keys: KeysCollection,
        label_key: KeysCollection,
        roi_scale: Union[Sequence[float], float],
        max_roi_scale: Union[Sequence[float], float, None] = None,
        random_center: bool = True,
        random_size: bool = False,
        image_threshold: float = 0.0,
        allow_smaller: bool = False,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        self.label_key = label_key
        cropper = RandScaleCropbyPos(
            roi_scale, 
            max_roi_scale, 
            random_center, 
            random_size, 
            image_threshold,
            allow_smaller=allow_smaller, 
            lazy=lazy
        )
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys, lazy=lazy)
        

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    CenterSpatialCropd,
    ToTensord,
)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
        global_crops_size: Tuple[float, float, float],
    ) -> Compose:
    transforms_list = [
        EnsureChannelFirstd(
            keys=["t1", "t1ce", "t2", "flair", "mask", "tumor_mask"],
            channel_dim="no_channel",
        ),
        CenterSpatialCropd(keys=["t1", "t1ce", "t2", "flair", "mask", "tumor_mask"], roi_size=global_crops_size),
        LastTransfromMRIs(gaussian_noise=False),
        ToTensord(keys=["transformed_image"], dtype=torch.float32),
    ]
    return Compose(transforms_list)
