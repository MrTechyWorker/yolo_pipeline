from interfaces.ingest_interface import IIngets_data
from loguru import logger
from pycocotools.coco import COCO

class DataIngester(IIngets_data):

    def get_coco_data(self, path: str) -> dict:
        """
        Args:
            COCO annotation path. : str

        Return:
            Returns a dicitonary with key values as id, dict of file, bbox, category_id respectively.

            Example:
            {1: {'file': 'annotated_images/2024-12-20_15-31-19.png',
                'bbox': [[473.1578947368421, 322.0, 176.8421052631578, 187.36842105263156]],
                'category_id': [0]},
                ...
            10: {'file': 'annotated_images/2024-12-20_15-34-27.png',
                'bbox': [[472.1052631578947, 329.36842105263156, 173.68421052631572, 182.1052631578948]],
                'category_id': [0]}
            }
        """
        try:
            self.coco_obj = COCO(path)
            logger.info(f"COCO handler created from {path}.")
        except Exception as e:
            logger.error(f"[⛔️] {e}")
    
        try:
            logger.info(f"[🏁] Fetching data from {path}.")
            _id_to_img: dict = {i:{"file": j["file_name"]} for i,j in self.coco_obj.imgs.items()}
            for i in _id_to_img:
                _id_to_img[i]["bbox"] = [self.coco_obj.anns[i]["bbox"]]
                _id_to_img[i]['category_id'] = [self.coco_obj.anns[i]["category_id"]]
            logger.success(f"[✅] Fetched data from {path}.")
            return _id_to_img
        except Exception as e:
            logger.error(f"[⛔️] {e}")
    
    def get_id_to_label(self) -> dict:
        """
        Args:
            -
        Return:
            Dict: Mapping of Ids and Labels in COCO dataset.
        """
        try:
            logger.info("[🏁] Mapping IDs to Category Names...")
            mapping = {i["id"]:i["name"] for i in self.coco_obj.cats.values()}
            logger.success("[✅] Mappings Collected.")
            return mapping
        except Exception as e:
            logger.error(f"[⛔️] {e}")