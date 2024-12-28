from abc import ABC, abstractmethod
import numpy 
import cv2
from loguru import logger
from utils.utils import yolotxt, coco_to_yolo


class ITransformer(ABC):

    @abstractmethod
    def transform(self, img_path, save_path): ...

class ITransformerBBox(ABC):
    
    @abstractmethod
    def transform_and_save(self, img_path, bbox, cat_id, out_path): ...


# ================================== Transformers Saving Interface =======================================


class TransformAndSaveBBOX:
    def transform_and_save(self, img_path , bbox, cat_id, out_path):
        out_path = out_path + self.id
        try:
            image = cv2.imread(img_path)
        except Exception as e:
            logger.error(f"[⛔️] {e}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h,w,_ = image.shape
        transformed = self.agument(image=image, bboxes=bbox, category_ids=cat_id)
        # logger.info(f"Image: {img_path} transformed.")
        normalised_bbox = coco_to_yolo(transformed["bboxes"], w,h)

        cv2.imwrite(out_path+".png", transformed['image'])
        yolotxt(out_path, transformed["category_ids"], normalised_bbox)

        return transformed['image']

class TransformAndSave:
    def transform(self, img_path, save_path):
        save_path = save_path+ self.id +".png"
        try:
            image = cv2.imread(img_path)
        except Exception as e:
            logger.error(f"[⛔️] {e}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.agument(image=image)
        cv2.imwrite(save_path, transformed['image'])
        return