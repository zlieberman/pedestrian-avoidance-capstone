import subprocess
import argparse
import glob
from shutil import copyfile


def fuse(vci, rgb_model, thermal_model, nir_model):

    copyfile(str(rgb_model) + '/best.pt', '/app/yolov3/rgb_model.pt')
    copyfile(str(thermal_model) + '/best.pt', '/app/yolov3/thermal_model.pt')
    copyfile(str(nir_model) + '/best.pt', '/app/yolov3/nir_model.pt')

    image = glob.glob(str(vci) + '/images/train/*.jpg')[0]

    subprocess.run(('python yolov3/detect.py --weights /app/yolov3/rgb_model.pt '
                    '/app/yolov3/thermal_model.pt /app/yolov3/nir_model.pt --source ' + image).split(' '))


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='fuse models')
    PARSER.add_argument('--video_color_images', type=str, help="video color images")
    PARSER.add_argument('--rgb_model', type=str, help="RGB model")
    PARSER.add_argument('--thermal_model', type=str, help="Thermal model")
    PARSER.add_argument('--nir_model', type=str, help="NIR model")

    ARGS = PARSER.parse_args()
    print('downloading images...')

    vci = ARGS.video_color_images
    rgb_model = ARGS.rgb_model
    thermal_model = ARGS.thermal_model
    nir_model = ARGS.nir_model

    fuse(vci, rgb_model, thermal_model, nir_model)
