# import package
import labelme2coco

# set directory that contains labelme annotations and image files
labelme_folder = "annotated_images_labelme"

# set export dir
export_dir = "data/"

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, export_dir)   