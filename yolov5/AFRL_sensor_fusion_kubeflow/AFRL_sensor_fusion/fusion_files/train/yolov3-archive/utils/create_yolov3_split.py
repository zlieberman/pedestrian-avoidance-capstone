from pathlib import Path
import numpy as np
import argparse

def main(args):
    root_path = Path(args.root_dir)
    ratio = args.ratio

    if not root_path.is_dir():
        raise NotADirectoryError()
    images_path = root_path/'images'
    n = len(list(images_path.iterdir()))

    dataset_name = root_path.name

    train_path = root_path/(dataset_name+'_train.txt')
    test_path = root_path/(dataset_name+'_test.txt')

    if args.random:
        i = 0
        u = np.random.uniform(size=n)
        assignment = u < ratio
        with train_path.open('w') as train, test_path.open('w') as test:
            for image in images_path.iterdir():
                img_string = './' + image.parent.name + '/' + image.name
                if assignment[i]:
                    train.write(img_string+'\n')
                else:
                    test.write(img_string+'\n')
                i += 1
    else:
        i = 0
        with train_path.open('w') as train, test_path.open('w') as test:
            for image in images_path.iterdir():
                img_string = './' + image.parent.name + '/' + image.name
                if i < n*ratio:
                    train.write(img_string+'\n')
                else:
                    test.write(img_string+'\n')
                i += 1
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root_dir', dest='root_dir',
                        help='root directory of the dataset', 
                        required=True, type=str)
    parser.add_argument('-t', '--train_ratio', dest='ratio',
                        help='ratio of training sets',
                        required=True, type=float)
    parser.add_argument('--random', dest='random',
                        help='randomize the data split',
                        action='store_true')
    args = parser.parse_args()
    main(args)