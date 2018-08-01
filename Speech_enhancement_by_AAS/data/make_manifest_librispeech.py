import csv
import os
from tqdm import trange
import torch
import numpy as np
import random
import pdb

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

# Basic config
prefix = '/home/kenkim/librispeech'
feat_prefix = prefix + '/feature'
txt_prefix = prefix + '/txt'
feat_dir_list = ['train_noisy_100000', 'val_noisy_5000', 'test_clean_5noises']
txt_dir_list = ['train', 'val', 'test_clean']
cleanfeat_dir_list = txt_dir_list
manifest_name_list = ['libri_tr_ny_paired.csv', 'libri_val_paired.csv', 'libri_te_paired.csv']

for i in range(len(manifest_name_list)):
    # Step 1) Gather all available files
    feat_dir = feat_prefix + '/' + feat_dir_list[i]
    clean_feat_prefix = feat_prefix + '/' + cleanfeat_dir_list[i]

    print('reading ' + feat_dir)
    noise_list = os.listdir(feat_dir)

    path_list = []
    T_list = []
    txt_list = []
    clean_list = []

    for j in range(len(noise_list)):
        snr_list = os.listdir(feat_dir + '/' + noise_list[j])
        for k in range(len(snr_list)):
            file_list = os.listdir(feat_dir + '/' + noise_list[j] + '/' + snr_list[k])
            print('noise = ' + noise_list[j] + ', snr = ' + snr_list[k])
            for l in range(len(file_list)):
                if(file_list[l][-1] == '7'): #pt7
                    clean_id = file_list[l].split('+')[0].split('.')[0]
                    feat_path = feat_dir + '/' + noise_list[j] + '/' + snr_list[k] + '/' + file_list[l]
                    feat = torch.load(feat_path)
                    txt_path = txt_prefix + '/' + txt_dir_list[i] + '/' + clean_id + '.txt'
                    clean_feat_path = clean_feat_prefix + '/'  + clean_id + '.pt7'
                    T = feat.size(1)

                    T_list.append(T)
                    path_list.append(feat_path)
                    txt_list.append(txt_path)
                    clean_list.append(clean_feat_path)

    # Step 2) Sort
    print('sorting ' + feat_dir)
    idx = sorted(range(len(T_list)), key=lambda k: T_list[k])

    # Write to manifest & Specify clean target
    manifest_name = manifest_name_list[i]
    print('write to manifest' + manifest_name)
    fp = open(manifest_name, 'w')
    csv_writer = csv.writer(fp)

    for j in trange(0, len(T_list)):
        line = [path_list[idx[j]], txt_list[idx[j]], clean_list[idx[j]]]  # input feature
        csv_writer.writerow(line)

    fp.close()


# Step 3) Convert train to train subset csv
print('step3) make training subset')
manifest_name_r = 'libri_tr_ny_paired.csv'
manifest_name_w = 'libri_trsub_ny_paired.csv'
nSample = 1000 # out of total(100000) samples

fp_r = open(manifest_name_r, 'r')
fp_w = open(manifest_name_w, 'w')
csv_writer = csv.writer(fp_w)

lines = fp_r.readlines()
lines_sample = random_combination(lines, nSample)
for line in lines_sample:
    line_splited = line.split(',')
    line_w = [line_splited[0], line_splited[1], line_splited[2][:-1]]
    #pdb.set_trace()
    csv_writer.writerow(line_w)

fp_r.close()
fp_w.close()



# Step 4) Convert paired to normal csv
print('step4) convert paired to normal csv')
manifest_name_list_r = ['libri_tr_ny_paired.csv', 'libri_trsub_ny_paired.csv', 'libri_val_paired.csv', 'libri_te_paired.csv']
manifest_name_list_w = ['libri_tr_ny.csv', 'libri_trsub_ny.csv', 'libri_val.csv', 'libri_te.csv']

for i in trange(0, len(manifest_name_list_r)):
    name_r = manifest_name_list_r[i]
    name_w = manifest_name_list_w[i]
    print(name_r + ' --> ' + name_w)

    fp_r = open(name_r, 'r')
    fp_w = open(name_w, 'w')
    csv_writer = csv.writer(fp_w)

    lines = fp_r.readlines()
    for line in lines:
        line_splited = line.split(',')
        #pdb.set_trace()
        #print(len(line_splited))
        line_w = [line_splited[0], line_splited[1]]
        csv_writer.writerow(line_w)

    fp_r.close()
    fp_w.close()