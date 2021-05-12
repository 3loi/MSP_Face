#!/usr/bin/env python
# coding: utf-8

# In[1]:
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
from glob import glob
from tqdm import tqdm
import argparse



argparse = argparse.ArgumentParser()
argparse.add_argument("-dir", "--dir", required=True)
args = vars(argparse.parse_args())
video_dir = args['dir']




def file_process(src_path):
    dst_directory_path = os.path.join(os.path.split(src_path)[0], 'jpg')

    if not os.path.exists(dst_directory_path):
        os.mkdir(dst_directory_path)
    if '.avi' not in src_path and '.mp4' not in src_path and '.flv' not in src_path:
        print("file not video", src_path)
        return -1
    

    #if destination file exists, delete it
    try:
        if os.path.exists(dst_directory_path):
            if not os.path.exists(os.path.join(dst_directory_path, 'image_00001.jpg')):
                subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
                print('remove {}'.format(dst_directory_path))
        else:
            os.mkdir(dst_directory_path)
    except:
        print(dst_directory_path)

    #extract the frames
    cmd = 'ffmpeg -i \"{}\" -qscale:v 8 \"{}/image_%05d.jpg\"'.format(src_path, dst_directory_path)
    
    #print(cmd)
    subprocess.call(cmd, shell=True)




extensions = ['.mp4']


video_dir = os.path.join(video_dir, '*', '*')

files = []
for ext in extensions:
    buf = os.path.join(video_dir, '*'+ext)
    print("Searching for files using {}".format(buf))
    files += glob(buf)

print("Found {} files with {} extensions".format(len(files), extensions))




for video in tqdm(files):
    filename = os.path.basename(video)
    cls = video.split(os.path.sep)[-2]
    file_process(video)


