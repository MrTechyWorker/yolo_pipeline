import threading

class Augmentor:
    def __init__(self, transformers_list: list):
        self.ts: list = transformers_list
    
    def __call__(self, path: str, save_path: str) -> None:
        """
        Iterates over list of transformers and runs each transformation on seperate thread.

        Args:
            - ima_path: str                 -> path to image to be augmented
            - out_path: str                 -> output path of image to save
        Return:
            None
        """
        for transformer in self.ts:
            # Create and start a daemon thread for each transformer
            thread = threading.Thread(target=transformer().transform, args=(path,save_path), daemon=True)
            thread.start()


class AugmentorBBox:
    def __init__(self, transformers_list: list):
        self.ts: list = transformers_list
    
    def __call__(self, img_path: str, bbox: list[list[float]], cat_id: list[int], out_path: str) -> None:
        """
        Iterates over list of transformers and runs each transformation on seperate thread.

        Args:
            - ima_path: str                 -> path to image to be augmented
            - bbox:     list[list[float]]   -> bbox params in coco formate from json in 2D array.
            - cat_id:   list[int]           -> category id of image
            - out_path: str                 -> output path of image to save
        Return:
            None
        """
        for transformer in self.ts:
            thread = threading.Thread(target=transformer().transform_and_save, args=(img_path, bbox, cat_id, out_path), daemon=True)
            thread.start()