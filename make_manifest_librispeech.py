import csv
import os
import fileinput
import shutil
import pdb

# Ver1. Create manifest files
"""
# Make manifest files for librispeech dataset
DB_name = 'librispeech'

# logSpec-Text manifest
feat_type = 'logSpec'

feat_dir = '/data3/kenkim/librispeech/log1plusspec'
feat_tr_dir = feat_dir + '/train'
feaet_val_dir = feat_dir + '/val'
feat_tecl_dir = feat_dir + '/test_clean'
feat_teot_dir = feat_dir + '/test_other'
dir_list=  [feat_tr_dir, feaet_val_dir, feat_tecl_dir, feat_teot_dir]

transcript_dir = 'data/LibriSpeech_dataset'
transcript_tr_dir = transcript_dir + '/train/txt'
transcript_val_dir = transcript_dir + '/val/txt'
transcript_tecl_dir = transcript_dir + '/test_clean/txt'
transcript_teot_dir = transcript_dir + '/test_other/txt'

transcript_dir_list = [transcript_tr_dir, transcript_val_dir, transcript_tecl_dir, transcript_teot_dir]


manifest_name = ['data/' + DB_name + '_manifest_' + feat_type + '_train.csv',
                 'data/' + DB_name + '_manifest_' + feat_type + '_valid.csv',
                 'data/' + DB_name + '_manifest_' + feat_type + '_test_clean.csv',
                 'data/' + DB_name + '_manifest_' + feat_type + '_test_other.csv']

for i in range(len(manifest_name)): # train/val/test
    csv_file = open(manifest_name[i], 'w')
    csv_writer = csv.writer(csv_file)

    transcript_file_list = os.listdir(transcript_dir_list[i])

    for j in range(len(transcript_file_list)):
        id = transcript_file_list[j].split('.')[0]
        print('i = ' + str(i+1) + '/' + str(len(manifest_name)) + ', j = ' + str(j) + '/' + str(len(transcript_file_list)) + ', filename = ' + id)

        line = [dir_list[i] + '/' + id + '.npy', transcript_dir_list[i] + '/' + id + '.txt']

        csv_writer.writerow(line)


    csv_file.close()



# logMel-Text manifest
feat_type = 'logMel'

feat_dir = '/data3/kenkim/librispeech/log1plusmel'
feat_tr_dir = feat_dir + '/train'
feaet_val_dir = feat_dir + '/val'
feat_tecl_dir = feat_dir + '/test_clean'
feat_teot_dir = feat_dir + '/test_other'
dir_list=  [feat_tr_dir, feaet_val_dir, feat_tecl_dir, feat_teot_dir]

transcript_dir = 'data/LibriSpeech_dataset'
transcript_tr_dir = transcript_dir + '/train/txt'
transcript_val_dir = transcript_dir + '/val/txt'
transcript_tecl_dir = transcript_dir + '/test_clean/txt'
transcript_teot_dir = transcript_dir + '/test_other/txt'

transcript_dir_list = [transcript_tr_dir, transcript_val_dir, transcript_tecl_dir, transcript_teot_dir]


manifest_name = ['data/' + DB_name + '_manifest_' + feat_type + '_train.csv',
                 'data/' + DB_name + '_manifest_' + feat_type + '_valid.csv',
                 'data/' + DB_name + '_manifest_' + feat_type + '_test_clean.csv',
                 'data/' + DB_name + '_manifest_' + feat_type + '_test_other.csv']

for i in range(len(manifest_name)): # train/val/test
    csv_file = open(manifest_name[i], 'w')
    csv_writer = csv.writer(csv_file)

    transcript_file_list = os.listdir(transcript_dir_list[i])

    for j in range(len(transcript_file_list)):
        id = transcript_file_list[j].split('.')[0]
        print('i = ' + str(i+1) + '/' + str(len(manifest_name)) + ', j = ' + str(j) + '/' + str(len(transcript_file_list)) + ', filename = ' + id)

        line = [dir_list[i] + '/' + id + '.npy', transcript_dir_list[i] + '/' + id + '.txt']

        csv_writer.writerow(line)


    csv_file.close()
"""

# Ver2. Convert audio manifest to feature manifest
audio_manifest_filepath_tr = 'data/libri_train_manifest.csv'
audio_manifest_filepath_val = 'data/libri_val_manifest.csv'
audio_manifest_filepath_tecl = 'data/libri_test_clean_manifest.csv'
audio_manifest_filepath_teot = 'data/libri_test_other_manifest.csv'
audio_manifest_list = [audio_manifest_filepath_tr, audio_manifest_filepath_val, audio_manifest_filepath_tecl, audio_manifest_filepath_teot]

# Type1. log1plusspec
feat_type = 'logSpec'
feat_manifest_filepath_tr = 'data/librispeech_' + feat_type + '_train_manifest.csv'; #copyfile(audio_manifest_filepath_tr, feat_manifest_filepath_tr)
feat_manifest_filepath_val = 'data/librispeech_' + feat_type + '_val_manifest.csv'; #copyfile(audio_manifest_filepath_val, feat_manifest_filepath_val)
feat_manifest_filepath_tecl = 'data/librispeech_' + feat_type + '_test_clean_manifest.csv'; #copyfile(audio_manifest_filepath_tecl, feat_manifest_filepath_tecl)
feat_manifest_filepath_teot = 'data/librispeech_' + feat_type + '_test_other_manifest.csv'; #copyfile(audio_manifest_filepath_teot, feat_manifest_filepath_teot)
feat_manifile_list = [feat_manifest_filepath_tr, feat_manifest_filepath_val, feat_manifest_filepath_tecl, feat_manifest_filepath_teot]


prefix_before = '/data3/kenkim/deepspeech.pytorch/data/LibriSpeech_dataset'
prefix_after = '/data3/kenkim/librispeech/feature/log1plusstft'
train_test_list= ['train', 'val', 'test_clean', 'test_other']

# Substitute rule
#1 directory : /data3/kenkim/deepspeech.pytorch/data/LibriSpeech_dataset/{train,val,test_clean, test_other}/wav -->  /data3/kenkim/feature/{log1plusspec, log1plusmel}/{train,val,test_clean, test_other}
#2 extension : .wav --> .npy

for i in range(len(train_test_list)):
    train_test = train_test_list[i]
    feat_manifest_file = feat_manifile_list[i]
    path_before = prefix_before + '/' + train_test + '/wav'
    path_after = prefix_after + '/' + train_test

    """
    # Substitute 1 : directory
    with fileinput.FileInput(feat_manifest_file, inplace=True) as file:
        for line in file:
            print(line.replace(path_before, path_after), end='')

    # Substitute 2 : extension
    """

    # Ver2
    # input file
    audio_manifest_file = audio_manifest_list[i]
    f = open(audio_manifest_file, 'r')
    filedata = f.read()
    f.close()
    #pdb.set_trace()

    # Substitute 1 : directory
    newdata = filedata.replace(path_before, path_after)

    # Substitute 2 : exntension
    newdata = newdata.replace('.wav', '.npy')

    # output file
    f = open(feat_manifest_file, 'w')
    f.write(newdata)
    f.close()

    print('convert & copy : ' + audio_manifest_file + ' --> ' + feat_manifest_file)


# Type2. log1plusmel
feat_type = 'logMel'
feat_manifest_filepath_tr = 'data/librispeech_' + feat_type + '_train_manifest.csv'; #copyfile(audio_manifest_filepath_tr, feat_manifest_filepath_tr)
feat_manifest_filepath_val = 'data/librispeech_' + feat_type + '_val_manifest.csv'; #copyfile(audio_manifest_filepath_val, feat_manifest_filepath_val)
feat_manifest_filepath_tecl = 'data/librispeech_' + feat_type + '_test_clean_manifest.csv'; #copyfile(audio_manifest_filepath_tecl, feat_manifest_filepath_tecl)
feat_manifest_filepath_teot = 'data/librispeech_' + feat_type + '_test_other_manifest.csv'; #copyfile(audio_manifest_filepath_teot, feat_manifest_filepath_teot)
feat_manifile_list = [feat_manifest_filepath_tr, feat_manifest_filepath_val, feat_manifest_filepath_tecl, feat_manifest_filepath_teot]


prefix_before = '/data3/kenkim/deepspeech.pytorch/data/LibriSpeech_dataset'
prefix_after = '/data3/kenkim/librispeech/feature/log1plusmel'
train_test_list= ['train', 'val', 'test_clean', 'test_other']

# Substitute rule
#1 directory : /data3/kenkim/deepspeech.pytorch/data/LibriSpeech_dataset/{train,val,test_clean, test_other}/wav -->  /data3/kenkim/feature/{log1plusspec, log1plusmel}/{train,val,test_clean, test_other}
#2 extension : .wav --> .npy

for i in range(len(train_test_list)):
    train_test = train_test_list[i]
    feat_manifest_file = feat_manifile_list[i]
    path_before = prefix_before + '/' + train_test + '/wav'
    path_after = prefix_after + '/' + train_test

    """
    # Substitute 1 : directory
    with fileinput.FileInput(feat_manifest_file, inplace=True) as file:
        for line in file:
            print(line.replace(path_before, path_after), end='')

    # Substitute 2 : extension
    """

    # Ver2
    # input file
    audio_manifest_file = audio_manifest_list[i]
    f = open(audio_manifest_file, 'r')
    filedata = f.read()
    f.close()
    #pdb.set_trace()

    # Substitute 1 : directory
    newdata = filedata.replace(path_before, path_after)

    # Substitute 2 : exntension
    newdata = newdata.replace('.wav', '.npy')

    # output file
    f = open(feat_manifest_file, 'w')
    f.write(newdata)
    f.close()

    print('convert & copy : ' + audio_manifest_file + ' --> ' + feat_manifest_file)


