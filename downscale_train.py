import os
import pathlib
import shutil
from tqdm import tqdm
import cv2
import random

s = '/mnt/person_data'
d = '/mnt/face_data/resize_02'
lines = open('train_full.txt').readlines()
lines_updated = []
random.shuffle(lines)
for l in tqdm(lines):
    data = l.strip().split()
    if 'person_data' not in l:
        continue
    file_path = data[0]
    dst_file = file_path.replace(s,d)
    rect_str,class_id = data[1].split(',')[:4],data[1].split(',')[-1]
    
    scale_size = 0.2

    res = list(map(lambda x: str(int(x*scale_size)), map(int, rect_str)))
    res.append(class_id)
    rect_str = ','.join(res)
    
    lines_updated.append('%s %s'%(dst_file, rect_str))

    if os.path.exists(dst_file):
        continue

    img = cv2.imread(file_path)
    resized = cv2.resize(img, (0,0), fx=scale_size, fy=scale_size, interpolation = cv2.INTER_AREA)

    dst_f = os.path.dirname(dst_file)
    pathlib.Path(dst_f).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(dst_file, resized)


with open('train_full_02.txt', 'w') as f:
    f.writelines(map(lambda s: s + '\n', lines_updated))















