import keras
import os
from PIL import Image

def cub_resize_save(data_path='D:\data\datasets\CUB_200_2011\images', width=224, height=224):
    
    save_dir = data_path+'_{}x{}'.format(width, height)
    print("resize CUB200 images to {}x{} and save them into {}".format(width, height, save_dir))
    train_dir = os.path.join(save_dir, 'train')
    val_dir = os.path.join(save_dir, 'val')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
        
    # label list from the data_path
    label_list = os.listdir(data_path)    
    for label in label_list:
        data_dir = os.path.join(data_path, label)
        file_list = os.listdir(data_dir)
        for idx, file_name in enumerate(file_list):
            file_path = os.path.join(data_dir, file_name)
            img = Image.open(file_path)
            img = img.resize((width, height))
            if idx<5:
                save_path = os.path.join(val_dir, label, file_name)
            else:
                save_path = os.path.join(train_dir, label, file_name)
                
            save_directory = os.path.dirname(save_path)
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            img.save(save_path)
    print("Finish resizing")
if __name__=='__main__':
    cub_resize_save()