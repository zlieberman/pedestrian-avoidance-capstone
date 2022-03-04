# Scripts

## hdf5 dataloader
`hdf5_dataloader.py` contains a dataloader object that converts `*.hdf5` files
into a directory in the correct format to be read by YOLO models.
The file structure is as follows:
```commandline
+-- files
|  +--images
|  |  +--train
|  |  |  +--train_image1.jpg
|  |  |  +--train_image2.jpg
|  |  +--val
|  |  |  +--val_image1.jpg
|  |  |  +--val_image2.jpg
|  +--labels
|  |  +--train
|  |  |  +--train_label1.txt
|  |  |  +--train_label2.txt
|  |  +--val
|  |  |  +--val_label1.txt
|  |  |  +--val_label2.txt
```

## Multispectral, NIR three-channel, Thermal, and VCI dataloaders
These dataloader scripts load image data from SICA-lab file share, either from
label files or from json files in the share itself.

## YOLOv3 onnx inference
`yolov3_onnx_inference.py` is a script for running inference on the MATLAB
version of YOLOv3. The MATLAB model must be exported to ONNX format, then it can be run using the script.