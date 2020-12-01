#!/usr/bin/env bash
# wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -xvf CUB_200_2011.tar
python write_CUB_filelist.py
