#!/usr/bin/env python
import os
import csv
import pdb

# train
input_file = 'chime3_manifest_mel_train.csv'
output_file = 'chime3_manifest_mel_train_clean_org_fix.csv'
prefix_w = '/data/kenkim/CHiME3/mel/singleCH/tr05_org'

fp_r = open(input_file, 'r')
lines = fp_r.readlines()

csv_file = open(output_file, 'w')
csv_writer = csv.writer(csv_file)

count = 0
for line in lines:
    count += 1
    splited_line = line.split(',')
    feat_path, transcript_path = splited_line[0], splited_line[1]
    transcript_path = transcript_path[:-1]
    feat_id = feat_path.split('/')[-1].split('.')[0].upper()
    feat_id_prefix = feat_id[0:3]
    feat_id_postfix = 'ORG'
    feat_id = feat_id_prefix + '_' + feat_id + '_' + feat_id_postfix

    feat_path_new = prefix_w + '/' + feat_id + '.npy'

    line = [feat_path_new, transcript_path]
    #pdb.set_trace()
    csv_writer.writerow(line)

fp_r.close()
csv_file.close()
