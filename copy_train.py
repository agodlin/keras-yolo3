import os
import pathlib
import shutil
from tqdm import tqdm

r = '/mnt/face_data/person_data'
d = '/mnt/face_data/person_data2'
lines = open('train.txt').readlines()

for l in tqdm(lines):
    l = l.strip().split()[0]
    if 'person_data' not in l:
        continue
    dst = l.replace(r,d)
    if os.path.exists(dst):
        continue
    dst_f = os.path.dirname(dst)
    pathlib.Path(dst_f).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(l,dst)
    















