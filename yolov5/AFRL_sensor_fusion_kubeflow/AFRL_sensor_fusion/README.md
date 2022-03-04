# AFRL Sensor Fusion
## Setup
### Enviroment Setup
#### Quick Setup
- install pip
```sudo apt-get install python3-pip```
- install dependency to run sensor fusion
```pip3 install -r env/requirements.txt```

#### Detailed Setup
**If you are not familliar with the setup please read the detailed setup process**

Refer to the [README](https://github.com/SICA-Lab/AFRL_sensor_fusion/blob/main/envs/README.md) file under [envs/](https://github.com/SICA-Lab/AFRL_sensor_fusion/tree/main/envs)

### Download pre-trained weight
- Download YOLOv3 pre-trained weight [here](https://northeastern-my.sharepoint.com/:u:/r/personal/j_martinez-lorenzo_northeastern_edu/Documents/SICA_LAB-DATA/pre-trained/YOLOv3%20Archive/yolov3.weights?csf=1&web=1&e=Rd1U4G)
  
  Store weight in `AFRL_sensor_fusion/detector/YOLOv3_archive/weight/`

- Download Monodepth2 weight [here](https://northeastern-my.sharepoint.com/:f:/r/personal/j_martinez-lorenzo_northeastern_edu/Documents/SICA_LAB-DATA/pre-trained/Monodepth2/mono+stereo?csf=1&web=1&e=RusmrR)
  
  Store the weight in `AFRL_sensor_fusion/depth_estimator/monodepth2/models/mono+stereo/`

- Download DeepSORT weight [here](https://northeastern-my.sharepoint.com/:u:/r/personal/j_martinez-lorenzo_northeastern_edu/Documents/SICA_LAB-DATA/pre-trained/DeepSORT/ckpt.t7?csf=1&web=1&e=KvaBGK)

  Store the weight in `AFRL_sensor_fusion/deep_sort/deep/checkpoint/`

**Note**: See all available ptr-traiend weights in [here](https://github.com/SICA-Lab/AFRL_sensor_fusion/blob/main/data/available_weight.md)

## Example
### Run on a video
To run the sensor fusion on a video (use drone detector)

```python3  yolov3_deepsort.py /PATH/TO/VIDEO --config_detection configs/yolov3-drone.yaml```

You can see the result stored in `output/`

## Training
Refer to the [README](https://github.com/SICA-Lab/AFRL_sensor_fusion/blob/main/train/README.md) file under [train/](https://github.com/SICA-Lab/AFRL_sensor_fusion/tree/main/train)

See all available dataset on SICA one drive [here](https://github.com/SICA-Lab/AFRL_sensor_fusion/blob/main/data/availabel_dataset.md)

## Build Custom Detector
Refer to the [README](https://github.com/SICA-Lab/AFRL_sensor_fusion/tree/main/detector/README.md) file under [detector/](https://github.com/SICA-Lab/AFRL_sensor_fusion/tree/main/detector)

## Build Custom Depth Estimator
Refer to the [README](https://github.com/SICA-Lab/AFRL_sensor_fusion/blob/main/depth_estimator/README.md) file under [depth_estimator/](https://github.com/SICA-Lab/AFRL_sensor_fusion/blob/main/depth_estimator/)

