import keras
import os

def cifar10_save(data_path = 'D:\data\cifar10', width=224, height=224):
    (x_train, y_train), (x_val, y_val) = keras.datasets.cifar10.load_data()

    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(val_path):
        os.makedirs(val_path)

    for i, x in enumerate(x_train):
        file_path = os.path.join(train_path, "{train_dir:02d}/{img_idx:05d}.jpg".format(train_dir=int(y_train[i]), img_idx=i))
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        img = keras.preprocessing.image.array_to_img(x)
        img = img.resize((width, height))
        img.save(file_path)


    for i, x in enumerate(x_val):
        file_path = os.path.join(val_path, "{val_dir:02d}/{img_idx:05d}.jpg".format(val_dir=int(y_val[i]), img_idx=i))
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        img = keras.preprocessing.image.array_to_img(x)
        img = img.resize((width, height))
        img.save(file_path)

if __name__=='__main__':
    cifar10_save(data_path = 'D:\data\cifar10_resize')