./fuse/build_and_push.sh
source ./NIR_3_channel_images/build_and_push.sh
source ./thermal_images/build_and_push.sh
source ./train_nir/build_and_push.sh
source ./train_rgb/build_and_push.sh
source ./train_thermal/build_and_push.sh
source ./video_color_images/build_and_push.sh
python pipeline.py