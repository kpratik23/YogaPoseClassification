import numpy as np
import cv2
import os
from tqdm import tqdm
import random

def random_rotation(image):
    
    k=random.randint(0, 3)
    image=np.rot90(image,k)
    return image

def random_crop(image,crop_ratio):
    
    
    crop_size=(int(image.shape[0]/crop_ratio),int(image.shape[1]/crop_ratio))
    
    assert image.shape[0] >= crop_size[0] and image.shape[1] >= crop_size[1],"Image dimensions should be larger than crop size"
    
    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2

    x = center_x - crop_size[1] // 2
    y = center_y - crop_size[0] // 2


    cropped_image=image[y:y + crop_size[0], x:x + crop_size[1]]
    cropped_image=cv2.resize(cropped_image,(image.shape[0],image.shape[1]))
    return cropped_image

def random_flip(image):
    
    flip=random.randint(0, 1)
    return cv2.flip(image, flip)


def apply_transforms(image):
    
    flipped_image=random_flip(image)
    rotated_image=random_rotation(image)
    cropped_image=random_crop(image,1.01)
    
    return [flipped_image,rotated_image,cropped_image]
    

def transform(testset):
    counter=0
    for classes in os.listdir(f"dataset/{testset}"):
        for im_path in tqdm(os.listdir(f"dataset/{testset}/{classes}/")):
            image=cv2.imread(f"dataset/{testset}/{classes}/{im_path}")
            transformed_images=apply_transforms(image)


            for i in range(len(transformed_images)):
                save_path=f"all_images/{testset}/{classes}/{counter}.jpg"
                cv2.imwrite(save_path, transformed_images[i])
                counter+=1

def delete_images(testset):
    for classes in os.listdir(f"dataset/{testset}"):
        for im_path in tqdm(os.listdir(f"dataset/{testset}/{classes}/")):
            os.remove(f"dataset/{testset}/{classes}/{im_path}")

transform(testset="train")
transform(testset="test")
