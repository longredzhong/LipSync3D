"""
Crop upper boddy in every video frame, square bounding box is averaged among all frames and fixed.
"""
import sys
import os
import cv2
import argparse
import math
from tqdm import tqdm
import torch
import utils
from utils import landmark_to_dict
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from audiodvp_utils import util
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
from multiprocessing import Pool


def get_reference_dict(data_dir):
    image = cv2.imread(os.path.join(data_dir, 'reference_frame.png'))
    image_rows, image_cols, _ = image.shape

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        reference_dict = landmark_to_dict(results.multi_face_landmarks[0].landmark)
        reference_dict = normalized_to_pixel_coordinates(reference_dict, image_cols, image_rows)
    return reference_dict

def draw_landmark(results, image, save_path):
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=results.multi_face_landmarks[0],
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())

    cv2.imwrite(save_path, image)


def normalized_to_pixel_coordinates(landmark_dict, image_width, image_height):
    def is_valid_normalized_value(value):
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    landmark_pixel_coord_dict = {}

    for idx, coord in landmark_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue

        if not (is_valid_normalized_value(coord[0]) and
                is_valid_normalized_value(coord[1])):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = coord[0] * image_width
        y_px = coord[1] * image_height
        z_px = coord[2] * image_width
        landmark_pixel_coord_dict[idx] = [x_px, y_px, z_px]
    return landmark_pixel_coord_dict


def draw_pose_normalized_mesh(target_dict, image, save_path):
    connections = mp_face_mesh.FACEMESH_TESSELATION
    drawing_spec = mp_drawing.DrawingSpec(color= mp_drawing.BLACK_COLOR, thickness=1, circle_radius=1)

    image_rows, image_cols, _ = image.shape
    R = target_dict['R']
    t = target_dict['t']
    c = target_dict['c']

    idx_to_coordinates = {}
    for idx, coord in target_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue
        tgt = np.array(coord).reshape(3, 1)
        norm_tgt = (c * np.matmul(R, tgt) + t).squeeze()
        x_px = min(math.floor(norm_tgt[0]), image_cols - 1)
        y_px = min(math.floor(norm_tgt[1]), image_rows - 1)
        landmark_px = (x_px, y_px)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    
    white_image = np.zeros([image_rows, image_cols, 3], dtype=np.uint8)
    white_image[:] = 255
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]

        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
            cv2.line(white_image, 
                idx_to_coordinates[start_idx],
                idx_to_coordinates[end_idx], 
                drawing_spec.color,
                drawing_spec.thickness
            )
    cv2.imwrite(save_path, white_image)


def draw_3d_mesh(target_dict, save_path, elevation=10, azimuth=10):
    connections = mp_face_mesh.FACEMESH_TESSELATION
    drawing_spec = mp_drawing.DrawingSpec(color= mp_drawing.BLACK_COLOR, thickness=1, circle_radius=1)

    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elevation, azim=azimuth)
    plotted_landmarks = {}

    for idx, coord in target_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue
        plotted_landmarks[idx] = (-coord[2], coord[0], -coord[1])

    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]

        if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
            landmark_pair = [plotted_landmarks[start_idx], plotted_landmarks[end_idx]]
            ax.plot3D(
                xs=[landmark_pair[0][0], landmark_pair[1][0]],
                ys=[landmark_pair[0][1], landmark_pair[1][1]],
                zs=[landmark_pair[0][2], landmark_pair[1][2]],
                color=(0., 0., 1.),
                linewidth=1)
    plt.savefig(save_path)

def multiProcess(im, data_dir, reference_dict):
    with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
        image = cv2.imread(im)
        annotated_image = image.copy()
        image_rows, image_cols, _ = image.shape
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        target_dict = landmark_to_dict(results.multi_face_landmarks[0].landmark)
        target_dict = normalized_to_pixel_coordinates(target_dict, image_cols, image_rows)
        R, t, c = utils.Umeyama_algorithm(reference_dict, target_dict)
        target_dict['R'] = R
        target_dict['t'] = t
        target_dict['c'] = c
        torch.save(target_dict, os.path.join(data_dir, 'mesh_dict', os.path.basename(im))[:-4]+'.pt')

        if args.draw_mesh:
            img_save_path = os.path.join(data_dir, 'mesh_image', os.path.basename(im)[:-4] + '.png')
            draw_landmark(results, annotated_image, img_save_path)

        if args.draw_norm_mesh:
            img_save_path = os.path.join(data_dir, 'mesh_norm_image', os.path.basename(im)[:-4] + '.png')
            draw_pose_normalized_mesh(target_dict, annotated_image, img_save_path)

        if args.draw_norm_3d_mesh:
            img_save_path = os.path.join(data_dir, 'mesh_norm_3d_image', os.path.basename(im)[:-4] + '.png')
            draw_3d_mesh(target_dict, img_save_path, elevation=10, azimuth=10)

def pose_normalization(args):
    data_dir = args.data_dir
    image_list = util.get_file_list(os.path.join(data_dir, 'crop'))
    reference_dict = get_reference_dict(data_dir)
    torch.save(reference_dict, os.path.join(data_dir, 'reference_mesh.pt'))

    data_dirs = []
    reference_dicts = []

    for i in range(len(image_list)):
        data_dirs.append(data_dir)
        reference_dicts.append(reference_dict)

    pool = Pool(processes=40)
    pool.starmap(multiProcess, zip(image_list, data_dirs, reference_dicts))

    # with mp_face_mesh.FaceMesh(
    # max_num_faces=1,
    # refine_landmarks=True,
    # min_detection_confidence=0.5,
    # min_tracking_confidence=0.5) as face_mesh:
        # for i in tqdm(range(len(image_list))):
        #     image = cv2.imread(image_list[i])
        #     annotated_image = image.copy()
        #     image_rows, image_cols, _ = image.shape
        #     results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #     target_dict = landmark_to_dict(results.multi_face_landmarks[0].landmark)
        #     target_dict = normalized_to_pixel_coordinates(target_dict, image_cols, image_rows)
        #     R, t, c = utils.Umeyama_algorithm(reference_dict, target_dict)
        #     target_dict['R'] = R
        #     target_dict['t'] = t
        #     target_dict['c'] = c
        #     torch.save(target_dict, os.path.join(data_dir, 'mesh_dict', os.path.basename(image_list[i]))[:-4]+'.pt')

        #     if args.draw_mesh:
        #         img_save_path = os.path.join(data_dir, 'mesh_image', os.path.basename(image_list[i])[:-4] + '.png')
        #         draw_landmark(results, annotated_image, img_save_path)

        #     if args.draw_norm_mesh:
        #         img_save_path = os.path.join(data_dir, 'mesh_norm_image', os.path.basename(image_list[i])[:-4] + '.png')
        #         draw_pose_normalized_mesh(target_dict, annotated_image, img_save_path)

        #     if args.draw_norm_3d_mesh:
        #         img_save_path = os.path.join(data_dir, 'mesh_norm_3d_image', os.path.basename(image_list[i])[:-4] + '.png')
        #         draw_3d_mesh(target_dict, img_save_path, elevation=10, azimuth=10)


def create_dirs(opt):
    os.makedirs(os.path.join(args.data_dir, 'mesh_dict'), exist_ok=True)
    if opt.draw_mesh:
        os.makedirs(os.path.join(args.data_dir, 'mesh_image'), exist_ok=True)
    
    if opt.draw_norm_mesh:
        os.makedirs(os.path.join(args.data_dir, 'mesh_norm_image'), exist_ok=True)

    if opt.draw_norm_3d_mesh:
        os.makedirs(os.path.join(args.data_dir, 'mesh_norm_3d_image'), exist_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--draw_mesh', type=bool, default=False)
    parser.add_argument('--draw_norm_mesh', type=bool, default=False)
    parser.add_argument('--draw_norm_3d_mesh', type=bool, default=False)
    args = parser.parse_args()

    create_dirs(args)
    pose_normalization(args)
