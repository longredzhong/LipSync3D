import sys
sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')

from audiodvp_utils import util
import numpy as np
import math
import torch
import os
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
from tqdm import tqdm
import cv2

# Input :
#       reference(dictionary from vertex idx to normalized landmark, dict[idx] = [x, y, z]) : landmark of reference frame.
#       target(dictionary from vertex idx to normalized landmark, dict[idx] = [x, y, z]) : landmark of target frame.
# Output : 
#       R : 3x3 Rotation matrix(np.array)
#       c : scale value(float)
#       t : 3x1 translation matrix(np.array)

def Umeyama_algorithm(reference, target):
    # idx 2 -> nose, 130 -> left eye, 359 -> right eye
    idx_list = [2, 94, 19, 1, 4, 5, 195, 197, 6, 168, 8, 9, 151, 10, 109, 108, 67, 69, 103, 104, 54, 68, 338, 337, 297, 299, 332, 333, 284, 298, 130, 243, 244, 359, 362, 463,
                21, 71, 162, 139, 156, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 301, 251, 55, 285, 193, 417, 122, 351, 196, 419, 3, 248, 51, 281,
                45, 275, 44, 274, 220, 440, 134, 363, 236, 456]
    # idx_list = [19, 243, 463]
    ref_points = []
    tgt_points = []

    for idx in idx_list:
        ref_points.append(reference[idx])
        tgt_points.append(target[idx])

    ref_points = np.array(ref_points)
    tgt_points = np.array(tgt_points)

    ref_mu = ref_points.mean(axis=0)
    tgt_mu = tgt_points.mean(axis=0)
    ref_var = ref_points.var(axis=0).sum()
    tgt_var = tgt_points.var(axis=0).sum()
    n, m = ref_points.shape
    covar = np.matmul((ref_points - ref_mu).T, tgt_points - tgt_mu) / n
    det_covar = np.linalg.det(covar)
    u, d, vh = np.linalg.svd(covar)
    detuv = np.linalg.det(u) * np.linalg.det(vh.T)
    cov_rank = np.linalg.matrix_rank(covar)
    S = np.identity(m)

    if cov_rank > m - 1:
        if det_covar < 0:
            S[m - 1, m - 1] = -1
    else: 
        if detuv < 0:
            S[m - 1, m - 1] = -1

    R = np.matmul(np.matmul(u, S), vh)
    c = (1 / tgt_var) * np.trace(np.matmul(np.diag(d), S))
    t = ref_mu.reshape(3, 1) - c * np.matmul(R, tgt_mu.reshape(3, 1))

    return R, t, c


def landmark_to_dict(landmark_list):
    landmark_dict = {}
    for idx, landmark in enumerate(landmark_list):
        landmark_dict[idx] = [landmark.x, landmark.y, landmark.z]

    return landmark_dict

def landmarkdict_to_normalized_mesh_tensor(landmark_dict):
    vertex_list = []
    for idx, coord in landmark_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue
        vertex_list.append(coord)
    
    if not ('R' in landmark_dict):
        return torch.tensor(vertex_list)
    
    R = torch.from_numpy(landmark_dict['R']).float()
    t = torch.from_numpy(landmark_dict['t']).float()
    c = float(landmark_dict['c'])
    vertices = torch.tensor(vertex_list).transpose(0, 1)
    norm_vertices = (c * torch.matmul(R, vertices) + t).transpose(0, 1)
    return norm_vertices


def landmarkdict_to_mesh_tensor(landmark_dict):
    vertex_list = []
    for idx, coord in landmark_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue
        vertex_list.append(coord)

    vertices = torch.tensor(vertex_list)
    return vertices

def mesh_tensor_to_landmarkdict(mesh_tensor):
    landmark_dict = {}
    for i in range(mesh_tensor.shape[0]):
        landmark_dict[i] = mesh_tensor[i].tolist()
    
    return landmark_dict


def draw_mesh_image(mesh_dict, save_path, image_rows, image_cols):
    connections = mp_face_mesh.FACEMESH_TESSELATION
    drawing_spec = mp_drawing.DrawingSpec(color= mp_drawing.BLACK_COLOR, thickness=1, circle_radius=1)

    idx_to_coordinates = {}
    for idx, coord in mesh_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue
        x_px = min(math.floor(coord[0]), image_cols - 1)
        y_px = min(math.floor(coord[1]), image_rows - 1)
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


def draw_mesh_images(mesh_dir, save_dir, image_rows, image_cols):
    mesh_filename_list = util.get_file_list(mesh_dir)

    for mesh_filename in tqdm(mesh_filename_list):
        mesh_dict = torch.load(mesh_filename)
        save_path = os.path.join(save_dir, os.path.basename(mesh_filename)[:-3] + '.png')
        draw_mesh_image(mesh_dict, save_path, image_rows, image_cols)
    
    return
