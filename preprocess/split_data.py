import os
import shutil
import random
import string
import argparse
from tqdm import tqdm
import pathlib
import pandas as pd

if __name__== "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=pathlib.Path, required=True, help='Path to input directory')
    parser.add_argument('--output_dir', type=pathlib.Path, required=True, help='Path to output directory')

    # Read arguments
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Make directories for train and test dataset
    train_dir = os.path.join(output_dir, 'train')
    train_img_dir = os.path.join(train_dir, 'images')
    train_mask_dir = os.path.join(train_dir, 'masks')
    test_dir = os.path.join(output_dir, 'test')
    test_img_dir = os.path.join(test_dir, 'images')
    test_mask_dir = os.path.join(test_dir, 'masks')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_mask_dir, exist_ok=True)

    # List images
    images_dir = os.path.join(input_dir, 'input_images')
    masks_dir = os.path.join(input_dir, 'masks')
    images = os.listdir(images_dir)

    # Rename files and split train and test
    original_fname = []
    encoded_fname = []
    isTrain = []

    characters = string.ascii_letters + string.digits
    random.seed(2022)

    for img in tqdm(images):
        # Make random name
        newid = ''.join(random.choice(characters) for i in range(10))
        while newid+'.png' in encoded_fname: # make sure random name doesn't exist
            newid = ''.join(random.choice(characters) for i in range(10))
        newid = newid+'.png'
        # Randomly select train or test
        random_number = random.uniform(0,1)
        if random_number < 0.2: #test
            imagedst = os.path.join(test_img_dir, newid)
            maskdst = os.path.join(test_mask_dir, newid)
            split=False
        else: #train
            imagedst = os.path.join(train_img_dir, newid)
            maskdst = os.path.join(train_mask_dir, newid)
            split=True
        # Move Files
        imagesrc = os.path.join(images_dir, img)
        masksrc = os.path.join(masks_dir, img)
        shutil.copy(imagesrc, imagedst)
        shutil.copy(masksrc, maskdst)
        # Store info
        original_fname.append(img)
        encoded_fname.append(newid)
        isTrain.append(split)

    # compile info as dataframe
    keydf = pd.DataFrame({'img_id':original_fname, 'encoded_id':encoded_fname, 'train':isTrain})
    keydf.to_csv(os.path.join(output_dir, 'keydf.csv'), index=False)