cp video_color_images.yaml yolov3/data/video_color_images.yaml

pip install -r requirements.txt

python video_color_images.py $1 $2

#pip install wandb[kubeflow]
#
#cd yolov3 || exit
#
#python train.py --img 256 --batch 32 --epochs 20 --weights yolov3.pt --data video_color_images.yaml  --nosave --workers 4