git clone https://github.com/ultralytics/yolov3  # clone repo
cp NIR.yaml yolov3/data/NIR.yaml
pip install wandb[kubeflow]
#python video_color_images.py

cd yolov3 || exit
pip install -r requirements.txt
mkdir labels_05_13_20_NIR1
cd labels_05_13_20_NIR1
mkdir images
cd images
mkdir train
mkdir val
cd ..
mkdir labels
cd labels
mkdir train
mkdir val
cd ..
cd ..

mkdir labels_05_13_20_NIR2
cd labels_05_13_20_NIR2
mkdir images
cd images
mkdir train
mkdir val
cd ..
mkdir labels
cd labels
mkdir train
mkdir val
cd ..
cd ..

mkdir labels_05_13_20_NIR3
cd labels_05_13_20_NIR3
mkdir images
cd images
mkdir train
mkdir val
cd ..
mkdir labels
cd labels
mkdir train
mkdir val
cd ..
cd ..


mkdir labels_05_13_20_NIR
cd labels_05_13_20_NIR
mkdir images
cd images
mkdir train
mkdir val
cd ..
mkdir labels
cd labels
mkdir train
mkdir val
cd ..
cd ..


mkdir labels_06_03_20_NIR
cd labels_06_03_20_NIR
mkdir images
cd images
mkdir train
mkdir val
cd ..
mkdir labels
cd labels
mkdir train
mkdir val