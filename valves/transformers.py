import albumentations as A 
from interfaces.transformer_interface import (ITransformer, TransformAndSave,
                                               ITransformerBBox, TransformAndSaveBBOX)


class Transformer1(TransformAndSave, ITransformer):
    def __init__(self):
        self.agument = A.Compose(
            [A.CenterCrop(height=480, width=480, p=1),
             A.HorizontalFlip(p=1)]
            )
        self.id = "1"
    
class Transformer2(TransformAndSave, ITransformer):
    def __init__(self):
        self.agument = A.Compose(
            [A.CenterCrop(height=480, width=480, p=1),
             A.VerticalFlip(p=1)]
            )
        self.id = "2"
    
class Transformer3(TransformAndSave, ITransformer):
    def __init__(self):
        self.agument = A.Compose(
            [A.HorizontalFlip(p=1)]
            )
        self.id = "3"

# ================================== BBOX Transformers =======================================

class Trasnformer1BBox(TransformAndSaveBBOX, ITransformerBBox):
    def __init__(self):
        self.agument = A.Compose(
            [A.HorizontalFlip(p=1)],
            bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
        )
        self.id = "1"

class Trasnformer2BBox(TransformAndSaveBBOX, ITransformerBBox):
    def __init__(self):
        self.agument = A.Compose(
            [A.CenterCrop(height=480, width=480, p=1)],
            bbox_params=A.BboxParams(format='coco', min_area=4500, label_fields=['category_ids']),
        )
        self.id = "2"
    
class Trasnformer3BBox(TransformAndSaveBBOX, ITransformerBBox):
    def __init__(self):
        self.agument = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
            ],
            bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
        )
        self.id = "3"
    
class Trasnformer4BBox(TransformAndSaveBBOX, ITransformerBBox):
    def __init__(self):
        self.agument = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.1, p=1.0),
                A.ShiftScaleRotate(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
            ],
            bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
        )
        self.id = "4"
    
class Trasnformer5BBox(TransformAndSaveBBOX, ITransformerBBox):
    def __init__(self):
        self.agument = A.Compose([
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
            ],
            bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
        )
        self.id = "5"
