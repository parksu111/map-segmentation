# Map Segmentation AI Competition
![alt text](https://github.com/parksu111/map-segmentation/blob/main/img/mask_ex.png)

## Introduction
This competition task was developed as part of an educational program hosted by [KOSMES](https://www.kosmes.or.kr/sbc/SH/MAP/SHMAP002M0.do) and run by [MNC](https://mnc.ai/). The actual competition page can be found [here](https://aiconnect.kr/competition/detail/220).

## Raw Data
The raw data used for this project can be downloaded [here](). It consists of:
* aerial images of Seoul and the Gyeonggi province in .tif format.
* json files containing polygon coordinates of different features within the images.
* json files containing meta data.

This [notebook]() gives a detailed overview of how the raw data was processed to prepare the train and test datasets for this competition.

## Task Overview
* Description: Determine whether each pixel in an aerial image corresponds to a building or not (Image segmentation).
* Data: PNG format aerial images and PNG format masks representing locations of buildings within the corresponding images.
  * $n_{train}$ = 10816
  * $n_{test}$ = 2704
* Evaluation: mIoU (mean intersection over Union)
  * The public leaderboard is calculated using 30% of the  test data. The final leader board is calculated using the public test set and the remaining private test set.
  
## Files
* *data_prep.ipynb* - Jupyter Notebook detailing how the raw data was processed to make the train and test datasets.
* *evaluation_prep.ipynb* - Jupyter Notebook detailing how evaluation of user predictions is done.
* preprocess/
  * *fix_utf.sh* - Bash script fixing utf-8 encoding error in raw json files.
  * *make_masks.py* - Python script to extract information from json files and make mask png files.
  * *split_data.py* - Python script to split train and test data.
* evaluate/
  * *evaluate.py* - File to read in participant predictions and calculate performance.
  * *answer.csv* - The actual building coordinates of the test images.
  * *sample_submission.csv* - Sample csv file that participants can use to prepare their prediction files.
