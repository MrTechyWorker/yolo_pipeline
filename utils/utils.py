import random
import shutil
import os
from loguru import logger
import yaml

def create_yaml_file(yaml_file_path: str, train_path: str, val_path: str, nc: int, labels: list, filename: str = 'data.yaml'):
    """
    Creates a YAML file for YOLO configuration.

    Args:
        file_path (str): Path to save the YAML file.
        train_path (str): Path to the training images folder.
        val_path (str): Path to the validation images folder.
        nc (int): Number of classes.
        names (list): List of class names.

    Return:
        - str: yaml file path
    """
    data = {
        "train": train_path,
        "val": val_path,
        "nc": nc,
        "names": labels,
    }

    path = yaml_file_path + '/' + filename
    
    with open(path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
    return path

def create_yolo_data_structure(image_folder_path, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    """
    Creates a YOLO data directory structure and segregates PNG and TXT files into train, valid, and test sets.

    Parameters:
    - train_ratio: Proportion of data for the training set.
    - valid_ratio: Proportion of data for the validation set.
    - test_ratio: Proportion of data for the test set.

    Return:
        tuple() ->  Training images stored path , 
                    Validation images stored path , 
                    Yolo Data structure path
    """
    # Get the current directory
    base_dir = os.getcwd() + '/' + image_folder_path
    logger.info(f"[#️⃣] Creating yolo file structure in: {base_dir}")
    
    # Create YOLO data structure
    yolo_data_dir = os.path.join(base_dir, "yolo_data")
    sub_dirs = ["train/images", "train/labels", "valid/images", "valid/labels", "test/images", "test/labels"]
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(yolo_data_dir, sub_dir), exist_ok=True)
        logger.info(f"[#️⃣] Created dir: {sub_dir}")
    
    # Get all PNG and TXT files
    png_files = sorted([f for f in os.listdir(base_dir) if f.endswith('.png')])
    txt_files = sorted([f for f in os.listdir(base_dir) if f.endswith('.txt')])
    
    # Ensure matching PNG and TXT files
    assert len(png_files) == len(txt_files), "Mismatch between PNG and TXT files"
    for png, txt in zip(png_files, txt_files):
        assert os.path.splitext(png)[0] == os.path.splitext(txt)[0], "File names do not match"

    # Shuffle the files
    combined_files = list(zip(png_files, txt_files))
    random.shuffle(combined_files)
    
    # Split the files into train, valid, and test sets
    total_files = len(combined_files)
    train_split = int(train_ratio * total_files)
    valid_split = int(valid_ratio * total_files) + train_split

    train_files = combined_files[:train_split]
    valid_files = combined_files[train_split:valid_split]
    test_files = combined_files[valid_split:]
    
    # Helper function to copy files
    def copy_files(file_list, dest_image_dir, dest_label_dir):
        for png, txt in file_list:
            shutil.copy(os.path.join(base_dir, png), os.path.join(dest_image_dir, png))
            shutil.copy(os.path.join(base_dir, txt), os.path.join(dest_label_dir, txt))
    
    # Copy files into respective directories
    copy_files(train_files, os.path.join(yolo_data_dir, "train/images"), os.path.join(yolo_data_dir, "train/labels"))
    copy_files(valid_files, os.path.join(yolo_data_dir, "valid/images"), os.path.join(yolo_data_dir, "valid/labels"))
    copy_files(test_files, os.path.join(yolo_data_dir, "test/images"), os.path.join(yolo_data_dir, "test/labels"))

    return yolo_data_dir + "/train/images", yolo_data_dir + "/valid/images", yolo_data_dir
    

def coco_to_yolo(bbox: list[list[float]], img_width, img_height):
    """
    Normlises coco bbox format from bbox_x, bbox_y, bbox_width and bbox_heigh 
    to yolo bbox normalised format x_center, y_center, bbox_width and bbox_xheigh.

    Return:
        list[list[fload]] -> Nomalised bbox dims

    """
    x, y, w, h = bbox[0]

    x_center = x + w / 2
    y_center = y + h / 2
    
    x_center /= img_width
    y_center /= img_height
    w /= img_width
    h /= img_height
    
    return [[x_center, y_center, w, h]]

def yolotxt(name: str, id: list, box: list[list]):
    """
    Saves a text file of image with bbox params, id in yolo format.
    """
    # Construct the YOLO format list of string/s
    data = [f"{int(id[i])} {' '.join(list(map(str, box[i])))}" for i in range(min(len(id), len(box)))]
    data_new = "\n".join(data)
    with open(name+".txt", "w") as f:
        f.write(data_new)