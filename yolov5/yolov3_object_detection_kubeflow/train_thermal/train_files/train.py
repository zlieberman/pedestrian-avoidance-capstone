import subprocess
import argparse


def train(thermal_images):
    with open('thermal_images.yaml', 'w+') as yaml:
        lines = [
        'train: ..' + str(thermal_images) + '/images/train/\n',
        'val: ..' + str(thermal_images) + '/images/val/\n',
        'nc: 1\n',
        'names: [\'drone\']\n',
        ]
        yaml.writelines(lines)
        print(lines)
    print('python yolov3/train.py --img 256 --batch 20 --epochs 20 --weights yolov3.pt --data thermal_images.yaml --nosave --workers 4'.split(' '))
    subprocess.run('python yolov3/train.py --img 256 --batch 20 --epochs 20 --weights yolov3.pt --data thermal_images.yaml --workers 4'.split(' '))
    subprocess.run(['echo', 'hello'])

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='download images')
    PARSER.add_argument('--thermal_images', type=str, help="thermal images")
    ARGS = PARSER.parse_args()
    print('downloading images...')

    thermal = ARGS.thermal_images
    train(thermal)

