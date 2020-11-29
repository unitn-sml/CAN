import numpy as np
from numpy import empty as np_empty, float32 as np_float32, sum as np_sum
import os
import sys
import random

from itertools import combinations, product, islice, combinations_with_replacement
from PIL import Image
from shutil import make_archive

# compatibility mode
from polygons_floor.experiment import load_experiment_config
from utils.utils_common import get_module_functions
from utils.utils_common import set_or_get_logger
import logging
import glob


def _read_binomial_dataset(files_names):
    images_data = []
    for image in files_names:
        images_data.append(np.array(Image.open(image).convert("L"))/255)
    return images_data


def check_parity_check(sample, pc_side):
    return np.array_equal(pc_side, np.mod(np_sum(sample, 1), 2))


def find_triangle_dimensions(area):
    width, total_area, height = 1, 1, 1
    while total_area < area:
        height+=1
        width+=2
        total_area+=width
    return width, height


def find_rhombus_dimensions(area):
    width, total_area, height = 1, 1, 1
    while total_area < area/2:
        height+=1
        width+=2
        total_area+=width
    return width, height * 2 - 1


def compute_rhombus_kernel(width_rhombus, height_rhombus):
    middle_width = np.ceil(width_rhombus / 2).astype(int) - 1
    rhombus_kernel = np.zeros(shape=(height_rhombus, width_rhombus))
    k= np.floor(width_rhombus / 2).astype(int) - 1
    for i in range(height_rhombus):
        for j in range(width_rhombus):
            if i < int(height_rhombus/2):
                if middle_width - i - 1 < j < middle_width + i + 1:
                    rhombus_kernel[i][j] = 1
                else:
                    rhombus_kernel[i][j] = 0
            if i == int(height_rhombus/2):
                rhombus_kernel[i][j] = 1
            if i > int(height_rhombus/2):
                if middle_width - k -1 < j < middle_width + k + 1:
                    rhombus_kernel[i][j] = 1
                else:
                    rhombus_kernel[i][j] = 0

        if i > int(height_rhombus / 2):
            k -= 1
    return rhombus_kernel



def find_polygons(sample, kernel):
    num_poly = 0
    polygons = []
    for i in range(0, sample.shape[0]-kernel.shape[0]+1):
        for j in range(0, sample.shape[1] - kernel.shape[1]+1):
            found = np.array_equal(sample[i:i + kernel.shape[0], j:j + kernel.shape[1]], kernel)
            if found:
                num_poly += int(found)
                polygons.append((i+1, j + 1, i + kernel.shape[0] + 1, j + kernel.shape[1] + 1))
    return num_poly, polygons


def extract_polygon(bboxes, kernel):
    polygon_pixels = [[(i, j) for i in range(bbox[0], bbox[2]) for j in
                range(bbox[1], bbox[3])] for bbox in bboxes]
    if len(polygon_pixels) != 0:
        polygons = polygon_pixels * np.expand_dims(kernel.flatten(), axis=[0,2])
        polygons = np.delete(polygons, np.where(polygons == (0,0)), axis=1)
        return [[(x[0], x[1]) for x in polygon] for polygon in polygons]
    else:
        return []


def polygon_generator_type0(top_y, top_x, img_width, img_height, area):
    # generate triangle
    if top_y >= img_height - 5 or top_x < 5 or top_x >= img_width - 5:
        return None
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


def polygon_generator_type1(top_y, top_x, img_width, img_height, area):
    # generate square
    square_side = int(np.sqrt(area))
    if top_x >= img_width - square_side or top_y >= img_height - square_side:
        return None
    pixels = [(top_y + i, top_x + j)
              for i, j in product(range(square_side), range(square_side))]
    assert len(pixels) == area
    return pixels


def polygon_generator_type2(top_y, top_x, img_width, img_height, area):
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
    if top_y >= img_height - num_row or \
            top_x < int(biggest_row / 2) + 1 or \
            top_x >= img_width - (int(biggest_row / 2) + 1):
        return None
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


def incompatible_polygons(polygons):
    def overlap(pol1, pol2):
        return any(pixel in pol2 for pixel in pol1)

    def touch(pol1, pol2):
        return any(abs(p1[0] - p2[0]) <= 1 and abs(p1[1] - p2[1]) <= 1
                   for p1, p2 in product(pol1, pol2))

    def incompatible(pol1, pol2):
        return overlap(pol1, pol2) or touch(pol1, pol2)

    return any(incompatible(c[0], c[1]) for c in combinations(polygons, 2))


def compute_log_likelihood_shit(experiment, samples):
    square_side = int(np.sqrt(experiment.area))
    square_kernel = np.asarray(
        [[1] * square_side for _ in range(square_side)])
    width_triangle, height_triangle = find_triangle_dimensions(experiment.area)
    middle_width = np.ceil(width_triangle / 2).astype(int) - 1

    triangle_kernel = np.asarray([[
                                      1 if middle_width - i - 1 < j < middle_width + i + 1 else 0
                                      for j in range(width_triangle)] for i in
                                  range(height_triangle)])

    width_rhombus, height_rhombus = find_rhombus_dimensions(experiment.area)
    rhombus_kernel = compute_rhombus_kernel(width_rhombus, height_rhombus)
    poly_counters = dict()
    for sample in samples:
        check_left = check_parity_check(sample[1:(experiment.img_height - 1), 1:int(experiment.img_width / 2)],
                                        sample[1:(experiment.img_height - 1), 0])
        check_right =\
            check_parity_check(sample[1:(experiment.img_height - 1),
                                      int(experiment.img_width / 2):experiment.img_width -1],
                               sample[1:(experiment.img_height - 1), experiment.img_width -1])
        target_area = experiment.polygons_number * experiment.area
        num_pixels = np_sum(sample[1:-1, 1:-1], axis=(0, 1)).astype(int)


        num_squares, squares_bboxes = find_polygons(np.array(sample[1:-1,1:-1]), square_kernel)
        num_triangles, triangles_bboxes = find_polygons(np.array(sample[1:-1, 1:-1]), triangle_kernel)
        num_rhombus, rhombus_bboxes = find_polygons(np.array(sample[1:-1, 1:-1]), rhombus_kernel)
        squares = extract_polygon(squares_bboxes, square_kernel)
        triangles = extract_polygon(triangles_bboxes, triangle_kernel)
        rhombus = extract_polygon(rhombus_bboxes, rhombus_kernel)
        polygons = squares + triangles + rhombus
        if not check_left or not check_right or not num_pixels == target_area or not len(polygons) == 2 or incompatible_polygons(polygons):
            continue
        else:
            update_dict(poly_counters, num_triangles, num_rhombus, num_squares)
    normalization_factor = sum(poly_counters[key] * experiment.polygons_prob[int(key[0])] * experiment.polygons_prob[int(key[1])] for key in poly_counters)
    probabilities = {key : poly_counters[key] * experiment.polygons_prob[int(key[0])] * experiment.polygons_prob[int(key[1])] / normalization_factor for key in poly_counters}
    log_likelihood = sum([np.log(value) for value in probabilities.values()])
    return log_likelihood


def compute_log_likelihood(experiment, fns, repetition):
    if repetition:
        possible_combinations =\
            list(combinations_with_replacement(["0", "1","2"], 2))
    else:
        possible_combinations = \
            list(combinations(["0", "1", "2"], 2))
    counter_dict = {}
    generated_images = {}
    test = set()
    possible_locations = list(product(range(1, experiment.img_height -1), range(1, experiment.img_width -1)))
    for i in possible_locations:
        for j in possible_locations:
            for x in possible_combinations:
                if x not in generated_images:
                    generated_images[x] = set()
                polygon_0 = [fns.get("polygon_generator_type%s" % s)
                        (i[0], i[1], experiment.img_width, experiment.img_height, experiment.area) for s in x[0]]
                polygon_1 = [fns.get("polygon_generator_type%s" % s)
                             (j[0], j[1], experiment.img_width,
                              experiment.img_height, experiment.area) for s in
                             x[1]]
                polygons = polygon_0 + polygon_1
                if None in polygons or incompatible_polygons(polygons):
                    continue
                signature = tuple([tuple(p) for p in sorted(polygons)])
                if signature not in generated_images[x]:
                    generated_images[x].add(signature)
                    test.add(signature)
                else:
                    continue
                if tuple(x) in counter_dict:
                    counter_dict[tuple(x)] += 1
                else:
                    counter_dict[tuple(x)] = 1
    normalization_factor = sum(counter_dict[key] * experiment.polygons_prob[int(key[0])] * experiment.polygons_prob[int(key[1])] for key in counter_dict)
    probabilities = {
        key: counter_dict[key] * experiment.polygons_prob[int(key[0])] *
             experiment.polygons_prob[int(key[1])] / normalization_factor for
        key in counter_dict}
    log_likelihood = sum([np.log(value) for value in probabilities.values()])
    return counter_dict, log_likelihood



def update_dict(poly_counters, num_triangles, num_rhombus, num_squares):
    signature = tuple(["0"] * num_squares +
                      ["1"] * num_triangles +
                      ["2"] * num_rhombus)
    if signature in poly_counters:
        poly_counters[signature] += 1
    else:
        poly_counters[signature] =1


def incompatible_polygons(polygons):
    def overlap(pol1, pol2):
        return any(pixel in pol2 for pixel in pol1)

    def touch(pol1, pol2):
        return any(abs(p1[0] - p2[0]) <= 1 and abs(p1[1] - p2[1]) <= 1
                   for p1, p2 in product(pol1, pol2))

    def incompatible(pol1, pol2):
        return overlap(pol1, pol2) or touch(pol1, pol2)

    return any(incompatible(c[0], c[1]) for c in combinations(polygons, 2))



def touch(polygon1, polygon2):
    for p1 in polygon1:
        for p2 in polygon2:
            if abs(p1[0] - p2[0]) <= 1 and abs(p1[1] - p2[1]) <= 1:
                return True
    return False


def main(experiment_path, repetition):
    logger = set_or_get_logger(__name__, None, logging.DEBUG, capacity=1)
    experiment = load_experiment_config(experiment_path)
    fns = get_module_functions(__name__)
    msg = "Going to evaluate log likelihood of samples {} ..."
    """files_names = glob.glob(glob.escape(sample_path) + "*.png")
    samples = _read_binomial_dataset(files_names)
    logger.info(msg.format(sample_path))"""
    counter, log_likelihood = compute_log_likelihood(experiment, fns, repetition)
    logger.info(f"Log Likelihood is {log_likelihood}")
    logger.info("Done!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", type=str, required=True,
                        help="Path to the experiment .json")
    parser.add_argument("-r", "--repetition", action="store_true",
                        help="Use repetition for combination")
    args = parser.parse_args()
    main(args.experiment, args.repetition)