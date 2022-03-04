import argparse
from tensorflow.keras.applications import VGG16
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


def train_model(data_split):
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    split = np.load(data_split, allow_pickle=True)
    (trainImages, testImages) = split[:2]
    (trainTargets, testTargets) = split[2:4]
    (trainFilenames, testFilenames) = split[4:]
    # load the VGG16 network, ensuring the head FC layers are left off
    vgg = VGG16(weights="imagenet", include_top=False,
                input_tensor=Input(shape=(224, 224, 3)))
    # freeze all VGG layers so they will *not* be updated during the
    # training process
    vgg.trainable = False
    # flatten the max-pooling output of VGG
    flatten = vgg.output
    flatten = Flatten()(flatten)
    # construct a fully-connected layer header to output the predicted
    # bounding box coordinates
    bboxHead = Dense(128, activation="relu")(flatten)
    bboxHead = Dense(64, activation="relu")(bboxHead)
    bboxHead = Dense(32, activation="relu")(bboxHead)
    bboxHead = Dense(4, activation="sigmoid")(bboxHead)
    # construct the model we will fine-tune for bounding box regression
    model = Model(inputs=vgg.input, outputs=bboxHead)

    # initialize the optimizer, compile the model, and show the model
    # summary
    INIT_LR = 1e-4
    NUM_EPOCHS = 25
    BATCH_SIZE = 32
    opt = Adam(lr=INIT_LR)
    model.compile(loss="mse", optimizer=opt)
    print(model.summary())
    # train the network for bounding box regression
    print("[INFO] training bounding box regressor...")
    H = model.fit(
        trainImages, trainTargets,
        validation_data=(testImages, testTargets),
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=1)
    MODEL_PATH = 'model.h5'
    PLOT_PATH = "plot.png"
    model.save(MODEL_PATH, save_format="h5")
    N = NUM_EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.title("Bounding Box Regression Loss on Training Set")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(PLOT_PATH)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split')
    args = parser.parse_args()
    train_model(args.data_split)
