import glob
from numpy.random import shuffle
import os
from PIL import Image
from shutil import make_archive


files_names = glob.glob(glob.escape("in/datasets/23k_20x20_S42_floor_planning/") + "*.png")
print(files_names[:10])

apt_images = []
res_images = []
double_res_images = []
for file_name in files_names:
    if "apt" in file_name:
        apt_images.append(file_name)
    elif "double_res" in file_name:
        double_res_images.append(file_name)
    else:
        res_images.append(file_name)
print(len(apt_images))
print(len(res_images))
print(len(double_res_images))
shuffle(apt_images)
shuffle(res_images)
shuffle(double_res_images)
apt_images_small = apt_images[:int(len(apt_images)*0.589)]
res_images_small = res_images[:int(len(res_images)*0.589)]
double_res_images_small = double_res_images[:int(len(double_res_images)*0.589)-1]
print(len(apt_images_small))
print(len(res_images_small))
print(len(double_res_images_small))
final_images = apt_images_small + res_images_small + double_res_images_small
shuffle(final_images)
new_folder = "in/datasets/13k_20x20_S42_floor_planning/"
os.makedirs(new_folder, exist_ok=True)

for i, file_name in enumerate(final_images, 0):
    image = Image.open(file_name)
    file_name_new = "img_{}_{}.png"
    if "apt" in file_name:
        image_filename = file_name_new.format(i, "apt")
    elif "double_res" in file_name:
        image_filename = file_name_new.format(i, "double_res")
    else:
        image_filename = file_name_new.format(i, "res")
    image.save(new_folder + image_filename)
datasets_folder = "in/datasets/"
make_archive(datasets_folder + "13k_20x20_S42_floor_planning", 'zip',
                 root_dir=new_folder)