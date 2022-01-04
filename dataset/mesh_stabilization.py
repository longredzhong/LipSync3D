import numpy as np
from one_euro_filter import OneEuroFilter
import os
from natsort import natsorted
import math
import cv2
import torch
from points import topology, mouthPoints, chins, rest

def applyFilter(points, t, min_cutoff, beta, skipPoints = []):
    filtered = np.empty_like(points)
    filtered[0] = points[0]
    one_euro_filter = OneEuroFilter(t[0], points[0], min_cutoff, beta)
    
    for i in range(1, points.shape[0]):
        filtered[i] = one_euro_filter(t[i], points[i])
        
    for i in range(1, points.shape[0]):
        for skipPoint in skipPoints:
            filtered[i, skipPoint] = points[i, skipPoint]

    return filtered

def draw_image(count, point):
    white_image = np.ones((256, 256, 3), np.uint8) * 255
    
    for start, end in topology:
        start_point = point[start,:2]
        end_point = point[end,:2]
        cv2.line(white_image, start_point.astype(int), end_point.astype(int), (0,0,0), 1)
        
    cv2.imwrite('test/{}.jpg'.format(count), white_image)

if __name__ == '__main__':
    image_height = 256
    image_width = 256

    normalised_mesh_files = natsorted([os.path.join('mesh_dict', x) for x in os.listdir(os.path.join('mesh_dict'))])
    landmarks = []
    for file in normalised_mesh_files:
        landmark = torch.load(file)
        R = landmark['R']
        t = landmark['t']
        c = landmark['c']
        keys = natsorted([x for x in landmark.keys() if type(x) is int])
        vertices = []
        for key in keys:
            vertice = np.array(landmark[key]).reshape(3,1)
            norm_vertice = (c * np.matmul(R, vertice) + t).squeeze()
            x_px = min(math.floor(norm_vertice[0]), image_width - 1)
            y_px = min(math.floor(norm_vertice[1]), image_height - 1)
            z_px = min(math.floor(norm_vertice[2]), image_width - 1)
            vertices.append([x_px, y_px, z_px])
        landmarks.append(vertices)
    
    landmarks = np.array(landmarks)
    
    shape_1, shape_2, shape_3 = landmarks.shape

    xs = landmarks[:,:,0].reshape((shape_1, shape_2))
    ys = landmarks[:,:,1].reshape((shape_1, shape_2))
    zs = landmarks[:,:,2].reshape((shape_1, shape_2))

    fps = 25
    t = np.linspace(0, xs.shape[0]/fps, xs.shape[0])

    xs_hat = applyFilter(xs, t, 0.005, 0.7)
    ys_hat = applyFilter(ys, t, 0.005, 0.7, mouthPoints + chins)
    ys_hat = applyFilter(ys_hat, t, 0.000001, 1.5, rest)
    zs_hat = applyFilter(zs, t, 0.005, 0.7)
    combine = np.stack(((xs_hat, ys_hat, zs_hat)), axis=2)

    count = [i for i in range(combine.shape[0])]

    os.makedirs(os.path.join('stabilized_norm_mesh'),exist_ok=True)
    for i in range(combine.shape[0]):
        torch.save(combine[i], os.path.join('stabilized_norm_mesh', '{}.pt'.format(count[i])))
