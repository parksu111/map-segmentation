import os 
import numpy as np
import json
import tifffile as tiff
import random
import matplotlib
import matplotlib.pyplot as plt
import cv2
import argparse
from tqdm import tqdm
import pathlib

# Function to transform polygon coordinates
def coord_to_points(coords, org_x, org_y, resolution):
    new_coord = []
    for point in coords:
        npoint = (max(0,point[0]-org_x), max(0,org_y-point[1]))
        new_coord.append(npoint)
    xs = [int(np.round(x[0]/resolution)) for x in new_coord]
    ys = [int(np.round(x[1]/resolution)) for x in new_coord]
    return xs,ys

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=pathlib.Path, required=True, help='Path to input directory')
    parser.add_argument('--output_dir', type=pathlib.Path, required=True, help='Path to output directory')

    # Read arguments
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Make directories for png image and mask
    image_dir = os.path.join(output_dir, 'input_images')
    mask_dir = os.path.join(output_dir, 'masks')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    ##### Make Masks #####
    # File directories
    rawtiffs_path = os.path.join(input_dir, 'raw_tiffs')
    labeljsons_path = os.path.join(input_dir, 'label_json')
    metafiles_path = os.path.join(input_dir, 'meta_fixed')

    # List files
    raw_tiffs = os.listdir(rawtiffs_path)
    label_jsons = os.listdir(labeljsons_path)
    meta_files = os.listdir(metafiles_path)

    # Find images that contain buildings
    contains_buildings = []
    for lbl in label_jsons:
        # load label json file
        lbl_path = os.path.join(labeljsons_path,lbl)
        label = json.load(open(lbl_path))
        # loop through featrues
        for feat in label['features']:
            if feat['properties']['ANN_CD']==10:
                img_name = lbl.split('_FGT.json')[0]
                contains_buildings.append(img_name)
                break

    # Loop through images and make masks
    for fid in tqdm(contains_buildings):
        # File names
        imfile = fid+'.tif'
        metafile = fid + '_META.json'
        labelfile = fid + '_FGT.json'
        # File paths
        impath = os.path.join(rawtiffs_path, imfile)
        metapath = os.path.join(metafiles_path, metafile)
        labelpath = os.path.join(labeljsons_path, labelfile)
        # Load files
        im = tiff.imread(impath)
        label_dict = json.load(open(labelpath))
        meta_dict = json.load(open(metapath))
        # Load coordinates for buildings
        building_coordinates = []
        for feat in label_dict['features']:
            if feat['properties']['ANN_CD']==10:
                coords = feat['geometry']['coordinates'][0]
                building_coordinates.append(coords)
        # Load META info
        meta_coord = meta_dict[0]['coordinates']
        mc_split = meta_coord.split(', ')
        org_x = float(mc_split[0])
        org_y = float(mc_split[1])
        img_res = meta_dict[0]['img_resolution']
        # Make and combine masks
        allmasks = []
        for sub_coord in building_coordinates:
            xs,ys = coord_to_points(sub_coord, org_x, org_y, img_res)
            points = np.array(list(zip(xs,ys)))
            maskim = np.zeros((512,512),dtype=np.int32)
            maskim = cv2.fillPoly(maskim, pts=[points], color=(255,255,255))
            allmasks.append(maskim)
        fin_mask = np.zeros((512,512), dtype=np.int32)
        for submask in allmasks:
            fin_mask = fin_mask + submask
        # Save Mask
        maskpath = os.path.join(mask_dir, fid+'.png')
        cv2.imwrite(maskpath, fin_mask)
        # Convert tif to png
        pngpath = os.path.join(image_dir, fid+'.png')
        cv2.imwrite(pngpath, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
