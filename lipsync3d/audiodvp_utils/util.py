import os
import pickle
from tqdm import tqdm
import cv2
from skimage import io
import torch
import numpy as np
import face_alignment
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from natsort import natsorted

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_file_list(data_dir, suffix=""):
    file_list = []

    for dirpath, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if suffix in filename:
                file_list.append(os.path.join(dirpath, filename))

    file_list = natsorted(file_list)

    return file_list


def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.

    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()

    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size))
        else:
            # raise KeyError('unexpected key "{}" in state_dict'.format(name))
            pass


def load_coef(data_dir, load_num=float('inf')):
    coef_list = []
    count = 0

    for filename in tqdm(get_file_list(data_dir)):
        coef = torch.load(filename)
        coef_list.append(coef)
        count += 1
        if count >= load_num:
            break

    return coef_list


def landmark_detection(image_list, save_path):
    fa_3d = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device='cuda')

    landmark_dict = {}

    for i in tqdm(range(len(image_list))):
        image_name = image_list[i]
        image = io.imread(image_name)
        preds = fa_3d.get_landmarks(image)

        assert preds is not None

        landmark_dict[image_name] = preds[0][:, :2]

    with open(save_path, 'wb') as f:
        pickle.dump(landmark_dict, f)


def plot_landmark(data_dir):
    create_dir(os.path.join(data_dir, 'landmark'))

    with open(os.path.join(data_dir, 'landmark.pkl'), 'rb') as f:
        landmark_dict = pickle.load(f)

    image_list = get_file_list(os.path.join(data_dir, 'crop'))

    for image_name in tqdm(image_list):
        image = cv2.imread(image_name)
        landmark = landmark_dict[image_name]

        for point in landmark:
            image = cv2.circle(image, (point[0], point[1]), radius=0, color=(255, 0, 0), thickness=-1)

        cv2.imwrite(os.path.join(data_dir, 'landmark', os.path.basename(image_name)), image)


def extract_face_emb(image_list, save_path, transforms_input):
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to('cuda')

    face_emb_dict = {}

    for i in tqdm(range(len(image_list))):
        image_name = image_list[i]
        image = Image.open(image_name).convert('RGB')

        input = transforms_input(image).to('cuda')
        input = input.reshape(1, 3, 224, 224)
        face_emb = facenet(input)

        face_emb_dict[image_name] = face_emb.squeeze().detach().to('cpu')
    
    with open(save_path, 'wb') as f:
        pickle.dump(face_emb_dict, f)


def load_face_emb(data_dir):
    face_emb_dir = os.path.join(data_dir, 'face_emb.pkl')

    with open(face_emb_dir, 'rb') as f:
        face_emb_dict = pickle.load(f)
    face_emb_list = list(face_emb_dict.values())
    return face_emb_list

def get_max_crop_region(crop_region_list):
    top, bottom, left, right = np.inf, 0, np.inf, 0

    for t, b, l, r in crop_region_list:
        if top > t:
            top = t

        if bottom < b:
            bottom = b
        
        if left > l:
            left = l
        
        if right < r:
            right = r
    
    return top, bottom, left, right