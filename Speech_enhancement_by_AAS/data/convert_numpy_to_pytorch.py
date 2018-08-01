import csv
import os
from tqdm import trange
import fileinput
import shutil
import pdb
import torch
import numpy as np

# Basic config
prefix = '/home/kenkim/librispeech'
feat_prefix = prefix + '/feature'
feat_dir_list = ['val', 'test_clean']

for i in range(len(feat_dir_list)):
    # Step 1) Gather all available files
    feat_dir = feat_prefix + '/' + feat_dir_list[i]
    print('reading ' + feat_dir)
    file_list = os.listdir(feat_dir)
    for j in trange(0, len(file_list)):
        if(file_list[j][-1] == 'y'):
            file_path_np = feat_dir + '/' + file_list[j]
            feat_np = np.load(file_path_np)
            feat_pt = torch.FloatTensor(feat_np)
            file_path_pt = file_path_np[:-3] + '.pt7'
            torch.save(feat_pt, file_path_pt)