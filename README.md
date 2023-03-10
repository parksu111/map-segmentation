# Map Segmentation AI Competition
![alt text](https://github.com/parksu111/map-segmentation/blob/main/img/mask_ex.png)

## Introduction
This competition task was developed as part of an educational program hosted by [KOSMES](https://www.kosmes.or.kr/sbc/SH/MAP/SHMAP002M0.do) and run by [MNC](https://mnc.ai/). The actual competition page can be found [here](https://aiconnect.kr/competition/detail/220).

This repository contains the code that went into the preparation of an AI competition task. This inlcudes:
* Processing raw data.
* Training a baseline model.
* Preparing code to evaluate user predictions.

## Competition Task Overview
* Description: Determine whether each pixel in an aerial image corresponds to a building or not (Image segmentation).
* Data: PNG format aerial images and PNG format masks representing locations of buildings within the corresponding images.
* Evaluation: mIoU (mean intersection over Union)
  * The public leaderboard is calculated using 30% of the  test data. The final leader board is calculated using the public test set and the remaining private test set.

## Raw Data
The raw data used for this project can be downloaded [here](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=143). It consists of:
* aerial images of Seoul and the Gyeonggi province in .tif format.
* json files containing polygon coordinates of different features within the images.
* json files containing meta data.

## Competition Preparation
* This [notebook](https://github.com/parksu111/map-segmentation/blob/main/data_prep.ipynb) contains EDA of the raw data along with a detailed explanation of the steps taken to process the raw data into data ready for a competition.
* This [notebook](https://github.com/parksu111/psg-classification/blob/main/evaluation_prep.ipynb) details the steps to prepare files/code used for evaluating user submissions.
* This [notebook](https://github.com/parksu111/psg-classification/blob/main/baseline.ipynb) contains the baseline model provided to participants.
  * The baseline model used for this task is the UNET (The encoder is the efficientnet-b0 pretrained on imagenet).

## Files
* *data_prep.ipynb* - Jupyter Notebook detailing how the raw data was processed to make the train and test datasets.
* *evaluation_prep.ipynb* - Jupyter Notebook detailing how evaluation of user predictions is done.
* *baseline.ipynb* - Jupyter Notebook of the baseline model for this task provided to students.
  * This notebook includes both the training and inference steps.
  * The same notebook with comments in Korean is also available.
* preprocess/
  * *fix_utf.sh* - Bash script fixing utf-8 encoding error in raw json files.
  * *make_masks.py* - Python script to extract information from json files and make mask png files.
  * *split_data.py* - Python script to split train and test data.
* evaluate/
  * *evaluate.py* - File to read in participant predictions and calculate performance.
  * *answer.csv* - The actual building coordinates of the test images.
  * *sample_submission.csv* - Sample csv file that participants can use to prepare their prediction files.
