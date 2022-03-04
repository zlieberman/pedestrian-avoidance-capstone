import subprocess
import argparse
import glob


def train(train, val):
    with open('NIR.yaml', 'w+') as yaml:
        lines = [
        'train: ..' + str(train) + '/images/train/\n',
        'val: ..' + str(val) + '/images/val/\n',
        'nc: 1\n',
        'names: [\'drone\']\n',
        ]
        yaml.writelines(lines)
        print(lines)
    print('python yolov3/train.py --img 256 --batch 8 --epochs 1 --weights yolov3.pt --data NIR.yaml --nosave --workers 4'.split(' '))
    subprocess.run('python yolov3/train.py --img 256 --batch 20 --epochs 20 --weights yolov3.pt --data NIR.yaml --workers 4'.split(' '))
    subprocess.run(['echo', 'hello'])
    # print(glob.glob('/**/*.pt', recursive=True))

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='download images')
    PARSER.add_argument('--nir_train_images', type=str, help="NIR training images")
    PARSER.add_argument('--nir_val_images', type=str, help="NIR validation images")
    ARGS = PARSER.parse_args()
    print('downloading images...')

    nir_train_images = ARGS.nir_train_images
    nir_val_images = ARGS.nir_val_images

    train(nir_train_images, nir_val_images)

