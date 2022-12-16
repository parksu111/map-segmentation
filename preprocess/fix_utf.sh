#!/bin/bash

cd /workspace/Competition/map_segmentation/data/RAW/train/meta
find -type f -exec iconv -f utf-8 -t utf-8 -c {} -o /workspace/Competition/map_segmentation/data/RAW/train/meta_fixed/{} \;