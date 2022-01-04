set -ex

# set data path
# target_dir : directory for training data
# source_dir : directory for inference data, put test audio in source_dir/audio directory
# video_dir : path for training video

target_dir="data/kkj/kkj04_lipsync3d"
source_dir="data/kkj/kkj04_lipsync3d"
video_dir="data/kkj/kkj04_lipsync3d/KKJ_slow_04_stand.mp4"


# set video clip duration
start_time="00:00:00"
end_time="240"

# mkdir -p $target_dir/full
# mkdir -p $target_dir/crop
# mkdir -p $target_dir/audio
# mkdir -p $target_dir/results
# mkdir -p $source_dir/audio
# mkdir -p $source_dir/results

# 1. Take all frames and audio of training data
# warning! the number of extracted frames should be dividable by 5. 
# If the number of frames of training video is not dividable by 5, delete some frames manually to make the number of frames dividable by 5

# ffmpeg -hide_banner -y -i $video_dir -r 25 $target_dir/full/%05d.png
# ffmpeg -hide_banner -y -i $video_dir -ar 16000 $target_dir/audio/audio.wav

# # crop and resize video frames
# python audiodvp_utils/crop_portrait.py \
#     --data_dir $target_dir \
#     --crop_level 1.5 \
#     --vertical_adjust 0.2

# pose normalization
# python lipsync3d/pose_normalization.py --data_dir $target_dir --gpu_ids 0

# train lipsync3d net
# python lipsync3d/train.py --src_dir $target_dir --tgt_dir $target_dir

# test lipsync3d net
python lipsync3d/test.py \
    --batch_size 1 \
    --serial_batches False \
    --isTrain False \
    --gpu_ids 0 \
    --src_dir $source_dir \
    --tgt_dir $target_dir

# ffmpeg -y -loglevel warning \
#     -thread_queue_size 8192 -i $target_dir/mesh_image/%05d.png \
#     -thread_queue_size 8192 -i $source_dir/reenact_mesh_image/%05d.png \
#     -i $source_dir/audio/audio.wav \
#     -filter_complex hstack=inputs=2 -shortest -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p $source_dir/results/tgt_kkj04_src_kkj00_mesh_reenact.mp4
