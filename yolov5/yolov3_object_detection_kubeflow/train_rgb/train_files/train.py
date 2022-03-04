import subprocess
import argparse
import glob


def train(video_color_images):
    with open('video_color_images.yaml', 'w+') as yaml:
        lines = [
        'train: ..' + str(video_color_images) + '/images/train/\n',
        'val: ..' + str(video_color_images) + '/images/val/\n',
        'nc: 1\n',
        'names: [\'drone\']\n',
        ]
        yaml.writelines(lines)
        print(lines)
    print('python yolov3/train.py --img 256 --batch 8 --epochs 20 --weights yolov3.pt --data video_color_images.yaml --nosave --workers 4'.split(' '))
    subprocess.run('python yolov3/train.py --img 256 --batch 20 --epochs 20 --weights yolov3.pt --data video_color_images.yaml --workers 4'.split(' '))
    # print(glob.glob('/app/**/*.pt', recursive=True))


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='download images')
    PARSER.add_argument('--video_color_images', type=str, help="video color images")
    ARGS = PARSER.parse_args()
    print('downloading images...')

    vci = ARGS.video_color_images
    train(vci)



