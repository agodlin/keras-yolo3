import os
import pathlib
import shutil
from tqdm import tqdm
import cv2

r = '/mnt/face_data/person_data'

lines = open('train.txt').readlines()
lines_updated = []
for l in tqdm(lines):
    data = l.strip().split()
    if 'person_data' not in l:
        continue
    file_path = data[0]
    dst_file = file_path.replace('person_data', 'person_data_05')
    rect,class_id = data[1].split(',')[:4],data[1].split(',')[-1]
    
    if os.path.exists(dst_file):
        continue
    
    dst_f = os.path.dirname(dst_file)
    pathlib.Path(dst_f).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(file_path,dst_file)
    















