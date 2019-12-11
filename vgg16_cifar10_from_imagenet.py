import argparse
import keras
import math
import os
import json
import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator

from cifar10_to_jpg import cifar10_save


def train(args):

    def lr_scheduler(epoch):
        if epoch < 10:
            lr = args.lr
        elif epoch < 20:
            lr = args.lr * 0.1
        elif epoch < 30:
            lr = args.lr * 0.01
        elif epoch < 40:
            lr = args.lr * 0.001
        return lr

    # Prepare data
    data_path = 'D:\data\cifar10_resize'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        cifar10_save(data_path=data_path)  # save cifar10 (32x32 -> 224x224)
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')

    train_generator = ImageDataGenerator(horizontal_flip=True,
                                         preprocessing_function=preprocess_input)
    train_gen = train_generator.flow_from_directory(train_path,
                                                    target_size=(224, 224),
                                                    batch_size=args.batch_size,
                                                    color_mode='rgb')

    val_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_gen = val_generator.flow_from_directory(val_path,
                                                  target_size=(224, 224),
                                                  batch_size=args.batch_size,
                                                  color_mode='rgb')
    #### VGGNet for Cifar10
    # 1. Build a model
    pre_model = VGG16(include_top=True, weights='imagenet')
    fc_output = pre_model.get_layer('fc2').output
    outputs = keras.layers.Dense(10, activation='softmax', name='predictions')(fc_output)
    model = keras.models.Model(inputs=pre_model.input, outputs=outputs)
    model.summary()
    # 2. Compile : optimizer, loss, metrics
    opt = keras.optimizers.SGD(lr=1e-3, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    # 3. Fit model : fit_generator / fit
    save_dir = 'D:\models\cifar10_vgg_from_imagenet'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # file_path = os.path.join(save_dir, 'weights.epoch.{epoch:03d}.val_acc.{val_acc:.4f}.h5')
    file_path = os.path.join(save_dir, 'best_model.h5')
    log_path = os.path.join(save_dir, 'log.csv')
    callbacks=[keras.callbacks.ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True),
               keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
               keras.callbacks.CSVLogger(log_path, append=True)]
    steps_per_epoch = int(math.ceil(train_gen.samples/args.batch_size))
    steps_val = int(math.ceil(val_gen.samples/args.batch_size))
    history = model.fit_generator(train_gen,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=args.epochs,
                                 validation_data=val_gen,
                                 validation_steps=steps_val,
                                 callbacks=callbacks)

    save_dir = "D://results\cifar10_vgg_from_imagenet"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(save_dir, 'accuracy.png'))
    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(save_dir, 'loss.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=4)
    args = parser.parse_args()
    train(args)
