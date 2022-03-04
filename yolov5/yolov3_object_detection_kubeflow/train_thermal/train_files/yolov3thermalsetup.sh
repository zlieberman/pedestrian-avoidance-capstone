git clone https://github.com/ultralytics/yolov3  # clone repo
cp thermal_images.yaml yolov3/data/thermal_images.yaml
pip install wandb[kubeflow]

cd yolov3 || exit
pip install -r requirements.txt
