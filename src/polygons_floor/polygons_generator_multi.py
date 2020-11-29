import numpy as np
import os
import sys
import random

from itertools import combinations, product
from PIL import Image
from shutil import make_archive

# compatibility mode
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from polygons_floor.experiment import load_experiment_config
from utils.utils_common import get_module_functions

from utils.utils_common import set_or_get_logger


def apply_parity_check(image, img_width, img_height, colours):
    assert img_width % 2 == 0
    assert img_height % 2 == 0

    white = (255, 255, 255)
    black = (0, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    red = (255, 255, 0)
    parity_colors = [black] + colours
    for j in range(1, img_height - 1):  # rows
        sum_left = sum(
            map(lambda pixel: pixel in colours,
                [image.getpixel((j, c)) for c in range(1, int(img_width / 2))]))
        sum_right = sum(
            map(lambda pixel: pixel in colours,
                [image.getpixel((j, c)) for c in range(int(img_width / 2),
                                                       img_width - 1)]))

        image.putpixel((j, 0), parity_colors[sum_left % len(parity_colors)])
        # if sum_left % 2 == 0:
        #    image.putpixel((j, 0), black)
        # else:
        #    image.putpixel((j, 0), white)

        image.putpixel((j, img_width - 1), parity_colors[sum_right % len(parity_colors)])

        # if sum_right % 2 == 0:
        #    image.putpixel((j, img_width - 1), black)
        # else:
        #    image.putpixel((j, img_width - 1), white)

    for j in range(1, img_width - 1):  # cols
        sum_top = sum(
            map(lambda pixel: pixel in colours,
                [image.getpixel((r, j)) for r in range(1,
                                                       int(img_height / 2))]))
        sum_bottom = sum(
            map(lambda pixel: pixel in colours,
                [image.getpixel((r, j)) for r in range(int(img_height / 2),
                                                       img_height - 1)]))

        image.putpixel((0, j), parity_colors[sum_top % len(parity_colors)])

        image.putpixel((img_height - 1, j),
                       parity_colors[sum_bottom % len(parity_colors)])
        # if sum_top % 2 == 0:
        #    image.putpixel((0, j), black)
        # else:
        #    image.putpixel((0, j), white)

        # if sum_bottom % 2 == 0:
        #    image.putpixel((img_height - 1, j), black)
        # else:
        #    image.putpixel((img_height - 1, j), white)


def generate_image(img_width, img_height, area, shapes, fns, generated_images,
                   parity_check, binomial, num_colors):
    # new image with black background
    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    colours_total = [white, red, green, blue]
    colours_number = 1 if binomial else num_colors
    colours = colours_total[0: min(colours_number, len(colours_total))]
    image = Image.new('RGB', (img_width, img_height), black)
    while True:
        possible_colours = list(colours)
        if isinstance(area, list):
            polygons = [fns.get("polygon_generator_type%s" % s)
                        (img_width, img_height, area[s]) for s in shapes]
        else:
            polygons = [fns.get("polygon_generator_type%s" % s)(img_width, img_height, area) for s in shapes]

        signature = tuple([tuple(p) for p in sorted(polygons)])
        if signature not in generated_images and \
                not incompatible_polygons(polygons):
            generated_images.add(signature)
            for polygon in polygons:
                pol_color = possible_colours.pop()
                for pixel in polygon:
                    image.putpixel((pixel[0], pixel[1]), pol_color)

            if parity_check:
                apply_parity_check(image, img_width, img_height, colours_total)
            break

    return image


def get_random_shapes(polygons_number, polygons_prob):
    random_output = np.random.multinomial(polygons_number, polygons_prob)
    polygon_types = []
    for j in range(len(random_output)):
        polygon_types += [j] * random_output[j]
    np.random.shuffle(polygon_types)
    return polygon_types


def incompatible_polygons(polygons):
    def overlap(pol1, pol2):
        return any(pixel in pol2 for pixel in pol1)

    def touch(pol1, pol2):
        return any(abs(p1[0] - p2[0]) <= 1 and abs(p1[1] - p2[1]) <= 1
                   for p1, p2 in product(pol1, pol2))

    def incompatible(pol1, pol2):
        return overlap(pol1, pol2) or touch(pol1, pol2)

    return any(incompatible(c[0], c[1]) for c in combinations(polygons, 2))


def polygon_generator_type0(img_width, img_height, area):
    # generate triangle

    top_x = int(np.random.uniform(5, img_width - 5))
    top_y = int(np.random.uniform(1, img_height - 5))
    last_layer = [(top_x, top_y)]
    pixels = last_layer
    while True:
        new_layer = [(x, y + 1) for (x, y) in last_layer]
        new_layer = [(last_layer[0][0] - 1, last_layer[0][1] + 1)] + new_layer
        new_layer.append((last_layer[-1][0] + 1, last_layer[-1][1] + 1))
        last_layer = new_layer
        pixels += last_layer
        if len(pixels) >= area:
            break
    assert len(pixels) == area
    return pixels


def polygon_generator_type1(img_width, img_height, area):
    # generate square
    square_side = int(np.sqrt(area))
    top_x = int(np.random.uniform(1, img_width - square_side))
    top_y = int(np.random.uniform(1, img_height - square_side))
    pixels = [(top_x + i, top_y + j)
              for i, j in product(range(square_side), range(square_side))]
    assert len(pixels) == area
    return pixels


def polygon_generator_type2(img_width, img_height, area):
    # generate rhombus
    remaining_area = area - 1
    pixel_counter = 1
    row = 1
    biggest_row = 1
    num_row = 1
    while remaining_area > 0 and row > 0:
        if pixel_counter < remaining_area:
            row = row + 2
            num_row = num_row + 1
            pixel_counter = pixel_counter + row
            remaining_area = remaining_area - row
            if row > biggest_row:
                biggest_row = row
        else:
            row = row - 2
            pixel_counter = pixel_counter + row
            num_row = num_row + 1
            remaining_area = remaining_area - row

    assert pixel_counter == area
    top_x = \
        int(np.random.uniform(
            int(biggest_row / 2) + 1, img_width - (int(biggest_row / 2) + 1)))
    top_y = int(np.random.uniform(1, img_height - num_row))
    last_layer = [(top_x, top_y)]
    pixels = last_layer
    symmetric = [(top_x, top_y + (num_row - 1))]
    for j in range(int(num_row / 2)):
        new_layer = [(x, y + 1) for (x, y) in last_layer]
        new_layer = [(last_layer[0][0] - 1, last_layer[0][1] + 1)] + new_layer
        new_layer.append((last_layer[-1][0] + 1, last_layer[-1][1] + 1))
        if j < int(num_row / 2) - 1:
            symmetric += \
                [(x, top_y + int(num_row - 2 - j)) for (x, y) in new_layer]
        last_layer = new_layer
        pixels += last_layer
    pixels += symmetric
    assert len(pixels) == area
    return pixels


def touch(polygon1, polygon2):
    for p1 in polygon1:
        for p2 in polygon2:
            if abs(p1[0] - p2[0]) <= 1 and abs(p1[1] - p2[1]) <= 1:
                return True
    return False


def main(experiment_path, parity_check):
    logger = set_or_get_logger(__name__, "generation_floor_planning.log")
    experiment = load_experiment_config(experiment_path)
    msg = "Going to generate {} images in {} folder..."
    logger.info(msg.format(experiment.dataset_size, experiment.dataset_folder))

    random.seed(experiment.dataset_seed)
    np.random.seed(experiment.dataset_seed)

    os.makedirs(experiment.dataset_folder, exist_ok=True)

    fns = get_module_functions(__name__)
    generated_images = set()

    for j in range(experiment.dataset_size):
        while True:
            shapes = get_random_shapes(experiment.polygons_number,
                                       experiment.polygons_prob)
            if len(shapes) == len(np.unique(shapes)):  # only different shapes
                break

        image = generate_image(experiment.img_width, experiment.img_height,
                               experiment.area, shapes, fns, generated_images,
                               parity_check, experiment.binomial, experiment.num_colors)

        # output format: img_SAMPLENUMBER_[POLYGONTYPES].png
        file_name = "img_{}_{}.png"
        image_filename = file_name.format(j, str(shapes).replace(" ", ","))
        image.save(experiment.dataset_folder + image_filename)
        if (j + 1) % 100 == 0:
            logger.info("{} images generated so far...".format(j + 1))

    logger.info("Compressing dataset in zip archive...")
    logger.info("Path: " + experiment.datasets_folder + experiment.dataset_name)
    make_archive(experiment.datasets_folder + experiment.dataset_name, 'zip',
                 root_dir=experiment.dataset_folder)
    logger.info("Done!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the experiment .json")
    parser.add_argument("-p", "--parity", type=bool, default=False,
                        help="Flag to add parity checker pixels")

    args = parser.parse_args()
    main(args.input, args.parity)
