import os
import cv2
from natsort import natsorted
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src_directory', type=str, required=True)

args = parser.parse_args()

if __name__ == '__main__':
    norm_images = natsorted([os.path.join(args.src_directory, 'reenact_mesh_image', x) for x in os.listdir(os.path.join(args.src_directory, 'reenact_mesh_image'))])
    out = cv2.VideoWriter('{}/temp_original.mp4'.format(args.src_directory), cv2.VideoWriter_fourcc(*'mp4v'), 25, (256, 256))

    for im in norm_images:
        image = cv2.imread(im)
        out.write(image)

    out.release()

    os.system('ffmpeg -y -i {}/temp_original.mp4 -i {}/audio/audio.wav -c:v copy -c:a aac {}/predicted_mesh.mp4'.format(args.src_directory, args.src_directory, args.src_directory))