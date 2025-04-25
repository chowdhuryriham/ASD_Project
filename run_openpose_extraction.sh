#!/bin/bash

# Run OpenPose to extract pose, hand, and face keypoints from a video

./build/examples/openpose/openpose.bin \
    --video "/home/riham/ASD_Project/data/videos/12.16.2024.mp4" \
    --model_pose BODY_25 \
    --hand --face \
    --scale_number 2 \
    --scale_gap 0.3 \
    --render_pose 1 \
    --display 0 \
    --write_json "/home/riham/ASD_Project/data/openpose_json" \
    --write_video "/home/riham/ASD_Project/data/videos/openpose_output1.avi" \
    --number_people_max -1 \
    --net_resolution "-1x288" \
    --hand_net_resolution "200x200" \
    --face_net_resolution "200x200" \
    --logging_level 3 \
    --write_keypoint_format json \
    --num_gpu_start 0 --num_gpu 1
