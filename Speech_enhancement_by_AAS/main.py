import torch
from config import get_config
from data_loader import DataLoader
import os
import json
#from utils import prepare_dirs_and_logger, save_config, check_config_used

def main(config):
    #prepare_dirs_and_logger(config)

    from data_loader import DataLoader
    if(config.trainer == 'minimize_DCE'):
        from trainer_DCE import Trainer
        paired = True
    elif(config.trainer == 'acoustic_supervision'):
        from trainer_acoustic import Trainer
        paired = False
    elif(config.trainer == 'AAS'):
        from trainer_AAS import Trainer
        paired = False
    #elif(config.trainer == 'AE'): # this mode is provided for data sanity check
#        from trainer_AE import Trainer
#        paired = True

    if config.gpu >= 0:
        torch.cuda.manual_seed(config.random_seed)
        torch.cuda.set_device(config.gpu)

    # Assign manifest
    if(config.DB_name == 'librispeech'):
        if(paired):
            config.tr_ny_manifest = 'data/libri_tr_ny_paired.csv'
            config.trsub_manifest = 'data/libri_trsub_ny_paired.csv'
            config.val_manifest = 'data/libri_val_paired.csv'
        else:
            config.tr_ny_manifest = 'data/libri_tr_ny.csv'
            config.trsub_manifest = 'data/libri_trsub_ny.csv'
            config.val_manifest = 'data/libri_val.csv'
        config.tr_cl_manifest = 'data/libri_tr_cl.csv'


    elif(config.DB_name == 'chime'):
        if(paired):
            config.tr_ny_manifest = 'data/chime_' + config.simul_real + '_tr_ny_paired.csv'
            config.trsub_manifest = 'data/chime_' + config.simul_real + '_trsub_ny_paired.csv'
            config.val_manifest = 'data/chime_real_val_paired.csv'
            confnig.val2_manifest = 'data/chime_simul_val_paired.csv'
        else:
            config.tr_ny_manifest = 'data/chime_' + config.simul_real + '_tr_ny.csv'
            config.trsub_manifest = 'data/chime_' + config.simul_real + '_trsub_ny.csv'
            config.val_manifest = 'data/chime_real_val.csv'
            confnig.val2_manifest = 'data/chime_simul_val.csv'
        config.tr_cl_manifest = 'data/chime_tr_org.csv' # it could be org or bth

    with open(config.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))

    data_loader = DataLoader(batch_size = config.batch_size, paired=paired,
                             tr_cl_manifest=config.tr_cl_manifest, tr_ny_manifest=config.tr_ny_manifest, trsub_manifest=config.trsub_manifest,
                             val_manifest=config.val_manifest, val2_manifest=config.val2_manifest, labels=labels)

    if not os.path.exists('logs/' + str(config.expnum)):
        os.makedirs('logs/' + str(config.expnum))

    trainer = Trainer(config, data_loader)

    torch.manual_seed(config.random_seed)



    if (config.mode == 'train'):
        trainer.train() # VAE
    elif(config.mode == 'test'):
        trainer.test()
    elif(config.mode == 'visualize'):
        trainer.visualize()


if __name__ == "__main__":
    config, unparsed = get_config()

    # check whether all the configuration is used in main.py & trainer.py
    #target_source = ['main.py', 'trainer.py', 'utils.py', 'trainer_supervised.py']
    #check_config_used(config, target_source)

    main(config)
