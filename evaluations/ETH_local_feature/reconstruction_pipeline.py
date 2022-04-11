# Import the features and matches into a COLMAP database.
#
# Copyright 2017: Johannes L. Schoenberger <jsch at inf.ethz.ch>

from __future__ import print_function, division

import os
import sys
import glob
import yaml
import types
import torch
import shutil
import sqlite3
import argparse
import subprocess
import multiprocessing

import numpy as np
from path import Path 
from tqdm import tqdm
import custom_matcher as cms


IS_PYTHON3 = sys.version_info[0] >= 3


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset_path", required=True,
    #                     help="Path to the dataset, e.g., path/to/Fountain")
    # parser.add_argument("--colmap_path", required=True,
    #                     help="Path to the COLMAP executable folder, e.g., "
    #                          "path/to/colmap/build/src/exe")
    # parser.add_argument("--features_path", required=True,
    #                     help="Path to the features folder, e.g., "
    #                          "path/to/feature")
    # parser.add_argument("--method_postfix", required=True,
    #                     help="the postfix of the method")
    # parser.add_argument("--matcher", required=True,
    #                     help="the matcher")
    parser.add_argument("--config", required=True,
                        help="Path to the configs, e.g., path/to/Fountain")
    args = parser.parse_args()
    return args


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2


def import_features_and_match(configs, paths):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    connection = sqlite3.connect(paths.database_path)
    cursor = connection.cursor()

    cursor.execute("SELECT name FROM sqlite_master "
                   "WHERE type='table' AND name='inlier_matches';")
    try:
        inlier_matches_table_exists = bool(next(cursor)[0])
    except StopIteration:
        inlier_matches_table_exists = False

    cursor.execute("DELETE FROM keypoints;")
    cursor.execute("DELETE FROM descriptors;")
    cursor.execute("DELETE FROM matches;")
    if inlier_matches_table_exists:
        cursor.execute("DELETE FROM inlier_matches;")
    else:
        cursor.execute("DELETE FROM two_view_geometries;")
    connection.commit()

    images = {}
    cursor.execute("SELECT name, image_id FROM images;")
    for row in cursor:
        images[row[0]] = row[1]

    for image_name, image_id in tqdm(images.items(), total=len(images.items())):
        feature_path = paths.features_path/'{}.{}'.format(image_name, configs['method_postfix'])
        feature_file = np.load(feature_path)

        keypoints = feature_file['keypoints'][:,:2]
        descriptors = feature_file['descriptors']
        assert keypoints.shape[1] == 2
        assert keypoints.shape[0] == descriptors.shape[0]

        keypoints_str = keypoints.tobytes() # early python3 use .tostring()
        cursor.execute("INSERT INTO keypoints(image_id, rows, cols, data) "
                       "VALUES(?, ?, ?, ?);",
                       (image_id, keypoints.shape[0], keypoints.shape[1],
                        keypoints_str))
        connection.commit()

    # custom match
    matcher = getattr(cms, configs['matcher'])
    image_names = list(images.keys())
    image_pairs = []
    image_pair_ids = set()
    for idx_total, image_name1 in enumerate(tqdm(image_names[:-1])):
        feature_path1 = paths.features_path/'{}.{}'.format(image_name1, configs['method_postfix'])
        descriptors1 = np.load(feature_path1)['descriptors']
        descriptors1 = torch.from_numpy(descriptors1).to(device)
        bar = tqdm(image_names[idx_total+1:])
        for idx_sub, image_name2 in enumerate(bar):
            image_pairs.append((image_name1, image_name2))
            image_id1, image_id2 = images[image_name1], images[image_name2]
            image_pair_id = image_ids_to_pair_id(image_id1, image_id2)
            if image_pair_id in image_pair_ids:
                continue

            feature_path2 = paths.features_path/'{}.{}'.format(image_name2, configs['method_postfix'])
            descriptors2 = np.load(feature_path2)['descriptors']
            descriptors2 = torch.from_numpy(descriptors2).to(device)

            matches = matcher(descriptors1, descriptors2, **configs['matcher_config'])
            assert matches.shape[1] == 2
            # bar.write("matches: {}".format(matches.shape[0]))
            image_pair_ids.add(image_pair_id)
            if image_id1 > image_id2:
                matches = matches[:, [1, 0]]

            matches_str = np.int32(matches).tostring()
            cursor.execute("INSERT INTO  matches(pair_id, rows, cols, data) "
                       "VALUES(?, ?, ?, ?);",
                       (image_pair_id, matches.shape[0], matches.shape[1],
                        matches_str))
            connection.commit()

    torch.cuda.empty_cache()
    with open(paths.match_list_path, 'w') as fid:
        for image_name1, image_name2 in image_pairs:
            fid.write("{} {}\n".format(image_name1, image_name2))
    cursor.close()
    connection.close()

    subprocess.call([paths.colmap_path,
                    "matches_importer",
                     "--database_path",
                     paths.database_path,
                     "--match_list_path",
                     paths.match_list_path,
                     "--match_type", "pairs"])

    # connection = sqlite3.connect(os.path.join(args.dataset_path, "database.db"))
    connection = sqlite3.connect(paths.database_path)
    cursor = connection.cursor()

    cursor.execute("SELECT count(*) FROM images;")
    num_images = next(cursor)[0]

    cursor.execute("SELECT count(*) FROM two_view_geometries WHERE rows > 0;")
    num_inlier_pairs = next(cursor)[0]

    cursor.execute("SELECT sum(rows) FROM two_view_geometries WHERE rows > 0;")
    num_inlier_matches = next(cursor)[0]

    cursor.close()
    connection.close()

    return dict(num_images=num_images,
                num_inlier_pairs=num_inlier_pairs,
                num_inlier_matches=num_inlier_matches)


def reconstruct(configs, paths):
    database_path = paths.database_path
    image_path = paths.image_path
    sparse_path = paths.features_path.parent/"{}_sparse".format(configs['subfolder'])
    dense_path = paths.features_path.parent/"{}_dense".format(configs['subfolder'])
    if not sparse_path.exists():
        sparse_path.makedirs_p()
    if not dense_path.exists():
        dense_path.makedirs_p()

    # Run the sparse reconstruction.
    subprocess.call([paths.colmap_path,
                    "mapper",
                    "--database_path", database_path,
                    "--image_path", image_path,
                    "--output_path", sparse_path,
                    "--Mapper.num_threads",
                    str(min(multiprocessing.cpu_count(), 16))])

    # Find the largest reconstructed sparse model.
    models = sparse_path.listdir()
    if len(models) == 0:
        print("Warning: Could not reconstruct any model")
        return

    largest_model = None
    largest_model_num_images = 0
    for model in models:
        subprocess.call([paths.colmap_path,
                        "model_converter",
                        "--input_path", model,
                        "--output_path", model,
                        "--output_type", "TXT"])
        with open("{}/cameras.txt".format(model), 'r') as fid:
            for line in fid:
                if line.startswith("# Number of cameras"):
                    num_images = int(line.split()[-1])
                    if num_images > largest_model_num_images:
                        largest_model = model
                        largest_model_num_images = num_images
                    break
    assert largest_model_num_images > 0

    # Run the dense reconstruction.
    largest_model_path = largest_model
    ### the codes for dense reconstruction
    # workspace_path = dense_path/largest_model.name
    # if not workspace_path.exists():
    #     workspace_path.makedirs_p()

    # subprocess.call([paths.colmap_path,
    #                  "image_undistorter",
    #                  "--image_path", image_path,
    #                  "--input_path", largest_model_path,
    #                  "--output_path", workspace_path,
    #                  "--max_image_size", "1200"])

    # subprocess.call([paths.colmap_path,
    #                  "patch_match_stereo",
    #                  "--workspace_path", workspace_path,
    #                  "--PatchMatchStereo.geom_consistency", "false"])

    # subprocess.call([paths.colmap_path,
    #                  "stereo_fusion",
    #                  "--workspace_path", workspace_path,
    #                  "--input_type", "photometric",
    #                  "--output_path", os.path.join(workspace_path, "fused.ply"),
    #                  "--StereoFusion.min_num_pixels", "5"])

    stats = subprocess.check_output(
        [paths.colmap_path, "model_analyzer",
         "--path", largest_model_path])

    stats = stats.decode().split("\n")
    for stat in stats:
        if stat.startswith("Registered images"):
            num_reg_images = int(stat.split()[-1])
        elif stat.startswith("Points"):
            num_sparse_points = int(stat.split()[-1])
        elif stat.startswith("Observations"):
            num_observations = int(stat.split()[-1])
        elif stat.startswith("Mean track length"):
            mean_track_length = float(stat.split()[-1])
        elif stat.startswith("Mean observations per image"):
            num_observations_per_image = float(stat.split()[-1])
        elif stat.startswith("Mean reprojection error"):
            mean_reproj_error = float(stat.split()[-1][:-2])

    # returns with dense results
    # with open(os.path.join(workspace_path, "fused.ply"), "rb") as fid:
    #     line = fid.readline().decode()
    #     while line:
    #         if line.startswith("element vertex"):
    #             num_dense_points = int(line.split()[-1])
    #             break
    #         line = fid.readline().decode()

    # return dict(num_reg_images=num_reg_images,
    #             num_sparse_points=num_sparse_points,
    #             num_observations=num_observations,
    #             mean_track_length=mean_track_length,
    #             num_observations_per_image=num_observations_per_image,
    #             mean_reproj_error=mean_reproj_error,
    #             num_dense_points=num_dense_points)

    ## returns without dense results
    return dict(num_reg_images=num_reg_images,
                num_sparse_points=num_sparse_points,
                num_observations=num_observations,
                mean_track_length=mean_track_length,
                num_observations_per_image=num_observations_per_image,
                mean_reproj_error=mean_reproj_error)


def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        configs['method_postfix'] = configs['postfix']
        configs['features_path'] = '../../ckpts/{}/desc'.format(configs['output_root'])
        configs['dataset_path'] = configs['data_config_extract']['data_path']
        configs['subfolder'] = configs['data_config_extract']['subfolder']

    paths = types.SimpleNamespace()
    paths.colmap_path = Path(configs['colmap_path'])/'colmap'
    paths.dataset_path = Path(configs['dataset_path'])/'{}'.format(
        configs['subfolder'])
    paths.image_path = paths.dataset_path/"images"

    paths.features_path = Path(configs['features_path'])/'{}'.format(
        configs['subfolder'])
    paths.database_path = paths.features_path.parent/'{}_{}.db'.format(
        configs['subfolder'], configs['method_postfix'])
    paths.match_list_path = paths.features_path/'image_pairs_{}.txt'.format(
        configs['method_postfix'])
    paths.result_path = Path(configs['features_path'])/'res_{}_{}.txt'.format(
        configs['subfolder'], configs['method_postfix'])

    # print(paths.match_list_path)
    if paths.database_path.exists():
        raise FileExistsError('The {} database already exists for method \
            {}.'.format(configs['subfolder'], configs['method_postfix']))
    shutil.copyfile(paths.dataset_path/'database.db', paths.database_path)

    matching_stats = import_features_and_match(configs, paths)
    reconstruction_stats = reconstruct(configs, paths)

    print()
    print(78 * "=")
    print("Raw statistics")
    print(78 * "=")
    print(matching_stats)
    print(reconstruction_stats)

    print()
    print(78 * "=")
    print("Formatted statistics")
    print(78 * "=")

    # strings = "| " + " | ".join(
    #         map(str, [paths.dataset_path.basename(),
    #                   "METHOD",
    #                   matching_stats["num_images"],
    #                   reconstruction_stats["num_reg_images"],
    #                   reconstruction_stats["num_sparse_points"],
    #                   reconstruction_stats["num_observations"],
    #                   reconstruction_stats["mean_track_length"],
    #                   reconstruction_stats["num_observations_per_image"],
    #                   reconstruction_stats["mean_reproj_error"],
    #                   reconstruction_stats["num_dense_points"],
    #                   "",
    #                   "",
    #                   "",
    #                   "",
    #                   matching_stats["num_inlier_pairs"],
    #                   matching_stats["num_inlier_matches"]])) + " |"

    strings_key = '{}|'.format(paths.dataset_path.basename())
    strings_val = '{}|'.format(paths.dataset_path.basename())
    for key, val in reconstruction_stats.items():
        strings_key += '{}|'.format(key)
        tmp_str = '{}'.format(val)
        tmp_str = tmp_str.rjust(len(key), ' ')
        tmp_str = tmp_str +'|'
        strings_val += tmp_str
    strings_key += '\n'
    strings_val += '\n'

    print(strings_key+strings_val)
    with open(paths.result_path, 'w') as fid:
        fid.write(strings_key+strings_val)


if __name__ == "__main__":
    main()
