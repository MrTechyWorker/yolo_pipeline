import time
from loguru import logger
import os
from valves.ingest_data import DataIngester
from valves.agumentor import Augmentor, AugmentorBBox
from utils.utils import create_yolo_data_structure, create_yaml_file
from valves.transformers import (
    Trasnformer1BBox, Trasnformer2BBox, Trasnformer3BBox, Trasnformer4BBox, Trasnformer5BBox,
    # Transformer1, Transformer2, Transformer3
    )

import warnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

# ================================== Config files =======================================

coco_data_set_path = "dataset.json"
out_image_folder = "imgs"

# =======================================================================================


# augmentor = Augmentor([Transformer1, Transformer2, Transformer3])

if __name__== "__main__":

    logger.info("[🏁] Starting pipline...")

    logger.info("[🏁] Setting up config files...")
    if not os.path.exists(out_image_folder):
        os.makedirs(out_image_folder)
        logger.info(f"[#️⃣] Folder created: {out_image_folder}")
    else:
        logger.info(f"[#️⃣] Folder exists: {out_image_folder}")

    if not os.path.isfile(coco_data_set_path):
        logger.error(f"[⛔️] File {coco_data_set_path} does not exist.")
        raise FileNotFoundError(f"File not found: {coco_data_set_path}")
    else:
        logger.info(f"[#️⃣] File exists: {coco_data_set_path}")
    logger.success("[✅] Config files initialized.")

    logger.info("[🏁] Initializing Data Ingester...")
    ingester = DataIngester()
    logger.success("[✅] Data Ingester initialized.")

    logger.info("[🏁] Fetching COCO data...")
    data = ingester.get_coco_data(coco_data_set_path)
    id_to_label = ingester.get_id_to_label()
    logger.success("[✅] COCO data fetched.")

    logger.info("[🏁] Initializing Augmentor...")
    bboxaugmentor = AugmentorBBox([Trasnformer1BBox, Trasnformer2BBox, Trasnformer3BBox, Trasnformer4BBox, Trasnformer5BBox])
    logger.success("[✅] Augmentor initialized.")

    logger.info("[🏁] Starting image augmentation...")
    ctr = 1
    for i,j in data.items():
        bboxaugmentor(j["file"], j["bbox"], j["category_id"],f"imgs/{ctr}")
        # augmentor(j["file"], f"imgs/{ctr}")
        ctr += 1
    logger.success("[✅] Augmentations done.")

    logger.info("[#️⃣] Waiting for threads to complete")
    time.sleep(2)

    logger.info("[🏁] Starting YOLO file structure...")
    yaml_train_path, yaml_valid_path, yolo_dir = create_yolo_data_structure("imgs")
    logger.success(f"[✅] YOLO file structure created in: {yolo_dir}.")

    logger.info("[🏁] Ctreating yaml file...")
    labels = [i[-1] for i in sorted(id_to_label.items())]
    num_cats = len(labels)
    yaml_file_path = create_yaml_file(yolo_dir, yaml_train_path, yaml_valid_path, num_cats, labels)
    logger.success(f"[✅] Yaml file created: {yaml_file_path}.")
    
    logger.success("[✅] Pipeline Successfully halted.")