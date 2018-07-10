import csv
import os

# Make manifest files for CHiME3 clean dataset

# Audio - Text
"""
chime3_clean_audio_dir = '/data3/kenkim/CHiME3/audio'
clean_audio_tr_dir = chime3_clean_audio_dir + '/tr05_orig_clean'
clean_audio_val_dir = chime3_clean_audio_dir + '/dt05_orig_clean'
clean_audio_te_dir = chime3_clean_audio_dir + '/et05_orig_clean'
audio_dir_list=  [clean_audio_tr_dir, clean_audio_val_dir, clean_audio_te_dir]

chime3_clean_transcript_dir = '/data3/kenkim/CHiME3/transcription'
clean_transcript_tr_dir = chime3_clean_transcript_dir + '/tr05_org_textonly' # simulated training data comes from WSJ0
clean_transcript_val_dir = chime3_clean_transcript_dir + '/dt05_bth_textonly' # simulated validation data comes from BTH
clean_transcript_te_dir = chime3_clean_transcript_dir + '/et05_bth_textonly' # simulated test data comes from BTH
transcript_dir_list = [clean_transcript_tr_dir, clean_transcript_val_dir, clean_transcript_te_dir]


manifest_name = ['data/chime3_manifest_audio_train.csv', 'data/chime3_manifest_audio_valid.csv', 'data/chime3_manifest_audio_test.csv']

for i in range(len(manifest_name)): # train/val/test
    csv_file = open(manifest_name[i], 'w')
    csv_writer = csv.writer(csv_file)

    transcript_file_list = os.listdir(transcript_dir_list[i])

    for j in range(len(transcript_file_list)):
        id = transcript_file_list[j].split('.')[0]
        print('i = ' + str(i) + '/' + str(len(manifest_name)) + ', j = ' + str(j) + '/' + str(len(transcript_file_list)) + ', filename = ' + id)

        line = [audio_dir_list[i] + '/' + id + '.wav', transcript_dir_list[i] + '/' + id + '.txt']

        csv_writer.writerow(line)


    csv_file.close()
"""


# Melspec(min-max normal)-Text manifest
chime3_clean_mel_dir = '/data3/kenkim/CHiME3/mel'
clean_mel_tr_dir = chime3_clean_mel_dir + '/tr05_orig_clean'
clean_mel_val_dir = chime3_clean_mel_dir + '/dt05_orig_clean'
clean_mel_te_dir = chime3_clean_mel_dir + '/et05_orig_clean'
mel_dir_list=  [clean_mel_tr_dir, clean_mel_val_dir, clean_mel_te_dir]

chime3_clean_transcript_dir = '/data3/kenkim/CHiME3/transcription'
clean_transcript_tr_dir = chime3_clean_transcript_dir + '/tr05_org_textonly' # simulated training data comes from WSJ0
clean_transcript_val_dir = chime3_clean_transcript_dir + '/dt05_bth_textonly' # simulated validation data comes from BTH
clean_transcript_te_dir = chime3_clean_transcript_dir + '/et05_bth_textonly' # simulated test data comes from BTH
transcript_dir_list = [clean_transcript_tr_dir, clean_transcript_val_dir, clean_transcript_te_dir]


manifest_name = ['data/chime3_manifest_mel_train.csv', 'data/chime3_manifest_mel_valid.csv', 'data/chime3_manifest_mel_test.csv']

for i in range(len(manifest_name)): # train/val/test
    csv_file = open(manifest_name[i], 'w')
    csv_writer = csv.writer(csv_file)

    transcript_file_list = os.listdir(transcript_dir_list[i])

    for j in range(len(transcript_file_list)):
        id = transcript_file_list[j].split('.')[0]
        print('i = ' + str(i) + '/' + str(len(manifest_name)) + ', j = ' + str(j) + '/' + str(len(transcript_file_list)) + ', filename = ' + id)

        line = [mel_dir_list[i] + '/' + id + '.npy', transcript_dir_list[i] + '/' + id + '.txt']

        csv_writer.writerow(line)


    csv_file.close()

