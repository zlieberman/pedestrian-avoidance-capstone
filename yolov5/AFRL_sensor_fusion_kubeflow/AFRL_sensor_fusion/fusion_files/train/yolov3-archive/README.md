# YOLOv3 DarkNet Version
## Setup
Follow instruction in [here](https://github.com/SICA-Lab/AFRL_sensor_fusion/tree/main/envs) to setup the python enviroment

## Avaiable Dataset

## Setup your dataset
**1. Data labeling** You can label your data using [Make Sense](https://www.makesense.ai/). You can use this tool on browser and it has option to export label in 
YOLO darknet format directly.

The label and image files should be under a parallel directory with the same name, for example:
```
data
|-images
  |-image1.jpg
  ...
|-labels
  |-image1.txt
  ...
```

The label is should be **0 indexed class follow with normalized xywh format**:

- One row per object
- Each row is `class` `x_center` `y_center` `width` `height` format.
- Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide x_center and width by image width, and y_center and height by image height.
- Class numbers are zero-indexed (start from 0).

For example: label of image contain 2 class 0
```
0 0.496261 0.365196 0.059829 0.048077
0 0.911592 0.264329 0.043269 0.032994
```

**2. create train/test split** You can create randomize train and test split using this [script](https://github.com/SICA-Lab/AFRL_sensor_fusion/blob/main/train/yolov3-archive/utils/create_yolov3_split.py)

```
cd utils
python3 create_yolov3_split.py -r /PATH/TO/DATASET/ -t [RATION OF TRAINING SET] --random
# example:
python3 create_yolov3_split.py -r ../data/drones/ -t 0.7 --random
```

You can also create these files manually, for eample:
```
# train.txt or test.txt
./images/drone1265.jpeg
./images/drone868.jpeg
./images/drone1378.jpg
```
The path should be the relative path to where `train.txt` and `test.txt` store

**3. Create new `*.names` file** This is the name of your classes, for example:
```
people
cars
bikes
cats
```

**4. Create new `*.data` file** This is your dataset descriptor, it should include the relative path (from where the training script is being call) to the test and train files and name files.
It should also include the number of classes. For example:
```
classes=1
train=data/drone_net/drone_net_train.txt
valid=data/drone_net/drone_net_test.txt
names=data/drone_net/labels.names
```

**5. Update cfg file** (optional). By default each YOLO layer has 255 outputs: 85 values per anchor [4 box coordinates + 1 object confidence + 80 class confidences], times 3 anchors. Update the settings to filters=[5 + n] * 3 and classes=n, where n is your class count. This modification should be made in all 3 YOLO layers.

## Training
Train the dataset by pass in the dataset descriptor file, configuration file, weights, and number of epochs.

If you want to train from the scratch use `''` as weights

Note: Use batch size 24 on SICA server (tested)

For example:

```python train.py --data data/drones.data --cfg cfg/yolov3-1cls.cfg --weights '' --epochs 1 --batch-size 24```

## Hyperparameter Tuning
You can pass `--evolve` to fine tuned the hyperparameter

The weight should be empty to train from scartch, and has only 1 epochs. It is set to 200 evolution by default

```python train.py --data data/drones.data --cfg cfg/yolov3-1cls.cfg --weights '' --epochs 1 --batch-size 24 --evolve```

You can check the result with generated `evolve.png` and `evolve.txt`

