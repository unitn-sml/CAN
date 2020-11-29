import pymzn
from PIL import Image
from itertools import product
from anytree import RenderTree
from anytree.importer import DictImporter
from anytree.iterators import LevelOrderIter
from shutil import make_archive
import random
import numpy as np
import os
from treeLayout import TreeSampler
import imgcluster
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from polygons_floor.experiment import load_experiment_config
from utils.utils_common import set_or_get_logger

logger = set_or_get_logger(__name__,  "generation_floor_planning.log")

room_type_dict = {
    "common_room": 1,
    "kitchen": 2,
    "bedroom": 3,
    "bathroom": 4,
    "corridor": 5,
    "livingroom": 6,
    "living": 7,
    "resting": 8,
    "door": 9
}

room_type_dict_color = {
    "common_room": (255, 255, 255),
    "kitchen": (255, 255, 0),
    "bedroom": (255, 0, 0),
    "bathroom": (0, 255, 255),
    "corridor": (0, 255, 0),
    "livingroom": (0, 0, 255),
    "living": (255,165,0),
    "resting": (255,0,255),
    "door": (128, 0, 0)
}

trees_per_type = {"apt": 0, "res": 0, "double_res": 0}
total_clusters = []

inverse_room_type_dict = {v: k for k, v in room_type_dict.items()}


def count_pixel(image):
    image_array = np.asarray(image)
    return image_array.any(axis=-1).sum()


def parse_house_tree(house_tree, building_type):

    nodes_info = [(node.id, node.parent.id, node.room_type)
                  if node.parent is not None
                  else (node.id, None, node.room_type)
                  for node in LevelOrderIter(house_tree)]
    signature = [building_type]
    for node in LevelOrderIter(house_tree):
        path = ""
        for n in node.path:
            path = path + n.room_type + "/"
        signature.append(path[:-1])
    signature.sort()
    rooms_number = len(nodes_info)
    rooms_type_id = [room_type_dict[room[2]] for room in nodes_info]
    rooms_adj = [[int(room1[0] == room2[1] or room2[0] == room1[1])
                  for room2 in nodes_info] for room1 in nodes_info]

    minizinc_params = {"ROOM_NUMBER": rooms_number,
                       "rooms_adj": rooms_adj,
                       "input_rooms_types": rooms_type_id}

    # print([nodes_info])
    # for room in rooms_adj:
    #    print(room)

    return minizinc_params, tuple(signature)


def sample_layout_tree(sampler):
    importer = DictImporter()
    building_type, house_tree = sampler.sampleInstance()
    tree = importer.import_(house_tree)
    print(str(RenderTree(tree)) + "\n", building_type)
    return tree, building_type


def _generate_image(processes, timeout, img_width, img_height, problem_path,
                    sampler, max_building_per_tree, generated_trees,
                    empty_trees, logger, dataset_size, apt_area, res_area,
                    double_res_area, images_per_cluster):
    while True:
        house_tree, building_type = sample_layout_tree(sampler)
        # a list of triples, each triple is composed of
        # (node id, node parent id, node room)

        minizinc_params, signature = parse_house_tree(house_tree, building_type)
        if building_type == "res":
            area_rooms = res_area
            num_solutions = max_building_per_tree * 4
        elif building_type == "apt":
            area_rooms = apt_area
            num_solutions = max_building_per_tree * 10
        elif building_type == "double_res":
            area_rooms = double_res_area
            num_solutions = max_building_per_tree

        wall_bx = [0 for _ in range(0,img_width - 2)] +\
                  [i for i in range(1,img_width - 1)] +\
                  [img_width - 1 for _ in range(0,img_width - 2)] +\
                  [i for i in range(1,img_width - 1)]

        wall_by = [i for i in range(2, img_width)] +\
                  [1 for _ in range(0, img_width - 2)] +\
                  [i for i in range(2, img_width)] +\
                  [img_width for _ in range(0, img_width - 2)]

        wall_tx = [1 for _ in range(0, img_width - 2)] +\
                  [i + 1 for i in range(1, img_width -1)] +\
                  [img_width for _ in range(0, img_width - 2)] +\
                  [i + 1 for i in range(1, img_width -1)]

        wall_ty = [i - 1 for i in range(2, img_width)] +\
                  [0 for _ in range(0, img_width - 2)] +\
                  [i - 1 for i in range(2, img_width)] +\
                  [img_width -1 for _ in range(0, img_width - 2)]

        wall_cells = len(wall_bx)
        minizinc_params = {
            **minizinc_params,
            "area_rooms": area_rooms,
            "SIDE": img_width,
            "wall_bx": wall_bx,
            "wall_by": wall_by,
            "wall_tx": wall_tx,
            "wall_ty": wall_ty,
            "WALL_CELLS": wall_cells}
        if signature not in generated_trees and signature not in empty_trees:
            generated_trees.add(signature)
            try:
                total_minizinc_buildings =\
                    pymzn.minizinc(
                        problem_path, data=minizinc_params, parallel=processes,
                        timeout=timeout, num_solutions=num_solutions)
                #selected_buildings =\
                #    random.sample(list(total_minizinc_buildings),
                #                  max_building_per_tree)
                np.random.shuffle(list(total_minizinc_buildings))
                selected_buildings = \
                    list(total_minizinc_buildings)[:max_building_per_tree]
            except (pymzn.MiniZincError,
                    pymzn.MiniZincError) as e:
                logger.info("Error, no rooms found")
                empty_trees.add(signature)
                continue
            images = []
            #TODO workaround, find why it does not work without range
            for i in range(len(selected_buildings)):
                building = selected_buildings[i]
                #print(rooms_minizinc)
                rooms = []
                door_dict =\
                    {key: list([value])
                     for key, value in building.items() if "door" in key}
                rooms_dict =\
                    {key: value
                     for key, value in building.items() if "room" in key}
                reshaped_rooms_list = list(zip(*list(rooms_dict.values())))
                reshaped_doors_list = list(zip(*list(door_dict.values())))
                for room_list in reshaped_rooms_list:
                    pixels = [(i, j)
                              for i, j in product
                              (range(room_list[0], room_list[2]),
                               range(room_list[3], room_list[1]))]
                    room = {"bottom_left": [(room_list[0], room_list[1])],
                            "top_right": [(room_list[2], room_list[3])],
                            "bottom_left_pixel":
                                [(room_list[0], room_list[1] - 1)],
                            "top_right_pixel":
                                [(room_list[2] - 1, room_list[3])],
                            "pixels": pixels,
                            "room_type": inverse_room_type_dict[room_list[4]]
                            }
                    rooms.append(room)

                for door_list in reshaped_doors_list:
                    pixels = [(i, j)
                              for i, j in product
                              (range(door_list[0], door_list[2]),
                               range(door_list[3], door_list[1]))]
                    room = {"bottom_left": [(door_list[0], door_list[1])],
                            "top_right": [(door_list[2], door_list[3])],
                            "bottom_left_pixel":
                                [(door_list[0], door_list[1] - 1)],
                            "top_right_pixel":
                                [(door_list[2] - 1, door_list[3])],
                            "pixels": pixels,
                            "room_type": "door"
                            }
                    rooms.append(room)

                # print(rooms)

                #for room in rooms:
                #    print(room)

                image = Image.new('RGB', (img_width, img_height), (0, 0, 0))

                for room in rooms:
                    for pixel in room["pixels"]:
                        image.putpixel((pixel[0], pixel[1]),
                                       room_type_dict_color[room["room_type"]])
                images.append((image, building_type))
            break
    clusters = imgcluster.do_cluster([image[0] for image in images],
                                    algorithm="SSIM", print_metrics=True)
    num_clusters = len(set(clusters)) if clusters is not None else 0
    final_images = []
    for cluster_index in range(0, num_clusters):
        cluster = [images[i] for i in range(0,len(clusters)) if clusters[i] == cluster_index]
        final_images.extend(cluster[:images_per_cluster])
    return final_images, num_clusters


def main(experiment_path, processes, timeout):
    experiment = load_experiment_config(experiment_path)
    msg = "Going to generate {} images in {} folder..."
    logger.info(msg.format(experiment.dataset_size, experiment.dataset_folder))
    pymzn.debug(False)

    random.seed(experiment.dataset_seed)
    np.random.seed(experiment.dataset_seed)

    sampler = TreeSampler(seedn=experiment.dataset_seed,
                          apt_people=experiment.apt_people,
                          res_people=experiment.res_people,
                          double_res_people=experiment.double_res_people,
                          buildings_prob=experiment.buildings_prob)
    generated_trees = set()
    empty_trees = set()
    houses_area = []
    os.makedirs(experiment.dataset_folder, exist_ok=True)
    try:
        assert experiment.img_height == experiment.img_width
    except AssertionError:
        logger.error("img_height and img_width must be equal")
    j = 0
    while j < experiment.dataset_size:
        images, num_clusters = \
            _generate_image(processes, timeout,
                            experiment.img_width, experiment.img_height,
                            experiment.problem_path, sampler,
                            experiment.max_building_per_tree,
                            generated_trees, empty_trees, logger,
                            experiment.dataset_size, experiment.apt_area,
                            experiment.res_area, experiment.double_res_area,
                            experiment.images_per_cluster)

        # output format: img_SAMPLENUMBER_[POLYGONTYPES].png
        file_name = "img_{}_{}.png"
        for image, building_type in images:
            if j < experiment.dataset_size:
                image_filename = file_name.format(j, building_type)
                image.save(experiment.dataset_folder + image_filename)
                if (j + 1) % 100 == 0:
                    logger\
                        .info("{} images generated so far..."
                              .format(j + 1))
                j = j + 1
                house_area = count_pixel(image)
                houses_area.append(house_area)
        if len(images) > 0:
            total_clusters.append(num_clusters)
            trees_per_type[images[0][1]] += 1
    cluster_mean = np.mean(total_clusters)
    cluster_dev_standard = np.std(total_clusters)
    area_mean = np.mean(houses_area)
    area_min = np.min(houses_area)
    area_max = np.max(houses_area)
    area_std = np.std(houses_area)
    logger.info("Metrics on cluster")
    logger.info("E[num_cluster] = {}".format(cluster_mean))
    logger.info("V[num_cluster] = {}".format(cluster_dev_standard))
    logger.info("trees per type: {}".format(str(trees_per_type)))
    logger.info("Metrics on area")
    logger.info("Number of houses = {}".format(len(houses_area)))
    logger.info("min[area] = {}".format(area_min))
    logger.info("max[area] = {}".format(area_max))
    logger.info("E[area] = {}".format(area_mean))
    logger.info("V[area] = {}".format(area_std))
    logger.info("Compressing dataset in zip archive...")
    make_archive(experiment.datasets_folder + experiment.dataset_name, 'zip',
                 root_dir=experiment.dataset_folder)
    logger.info("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the experiment .json")
    parser.add_argument("-p", "--processes", type=int, required=False,
                        help="Number of process for minizinc task", default=4)
    parser.add_argument("-t", "--timeout", type=int, required=False,
                        help="Timeout in millis for minizinc solver",
                        default=10)

    args = parser.parse_args()
    main(args.input, args.processes, args.timeout)
