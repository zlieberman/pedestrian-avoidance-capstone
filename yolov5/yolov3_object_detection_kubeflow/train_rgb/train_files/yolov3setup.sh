git clone https://github.com/ultralytics/yolov3  # clone repo
cp video_color_images.yaml yolov3/data/video_color_images.yaml
pip install wandb[kubeflow]
#python video_color_images.py

cd yolov3 || exit
pip install -r requirements.txt
