import cv2
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--target_video', type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":
    os.makedirs('full', exist_ok=True)
    cap = cv2.VideoCapture(args.target_video)
    count = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join('full', '{}.png'.format(count)), frame)
            count += 1
        else:
            break

    cap.release()
