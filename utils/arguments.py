import argparse
import os
import shutil
from pathlib import Path
import yaml


class Configs:
    def __init__(self):
        self.REQUIRE_CONFGS = None
        self.cfgs = self.__INIT_CONFIGS()
        assert self.REQUIRE_CONFGS is not None
        
        self.__PREPROCESS()


    def __INIT_CONFIGS(self):
        parser = argparse.ArgumentParser()

        self.REQUIRE_CONFGS = ['epoch', 'batch_size', 'learning_rate', 'weight_decay']

        # global setting
        parser.add_argument('-te', '--testing', 
                            default=False, action='store_true', 
                            help='When this arugument is provided, the running mode will be set to test')
        # training setting
        parser.add_argument('-cp', '--checkpoint_path', 
                            type=str, default='checkpoints/tmp/', 
                            help='Model checkpoint path [default: checkpoints/tmp/]')
        parser.add_argument('-re', '--resume', 
                            default=False, action='store_true',
                            help='When this arugument is provided, the program will resume from checkpoint file (only in training mode)')
        parser.add_argument('-ep', '--epoch', 
                            type=int, default=5, 
                            help='Epoch to run [default: 5]')
        parser.add_argument('-bs', '--batch_size', 
                            type=int, default=16, 
                            help='Batch Size during training [default: 16]')
        parser.add_argument('-lr', '--learning_rate', 
                            type=float, default=0.001, 
                            help='Initial learning rate [default: 0.001]')
        parser.add_argument('-wd', '--weight_decay', 
                            type=float, default=0, 
                            help='Optimization L2 weight decay [default: 0]')
        # testing setting
        parser.add_argument('-rp', '--result_path', 
                            type=str, default='resullt/tmp/', 
                            help='Outputs save path [default: resullt/tmp/]')
        
        return parser.parse_args()
    
    def __PREPROCESS(self):
        if self.cfgs.resume or self.cfgs.testing:
            self.__LOAD_CHECKPOINT_CONFIGS()
        else:
            self.__CLEAR_CHECKPOINT_PATH_FILE()
            self.__SAVE_CONFIGS()

    def __LOAD_CHECKPOINT_CONFIGS(self):
        with open(self.cfgs.checkpoint_path + 'configs.yml', 'r') as yaml_file:
            loaded_cfgs = yaml.safe_load(yaml_file)

        for key, value in loaded_cfgs.items():
            if key in self.cfgs.__dict__:
                self.cfgs.__dict__[key] = value

    def __CLEAR_CHECKPOINT_PATH_FILE(self):
        for elm in Path(self.cfgs.checkpoint_path).glob('*'):
            elm.unlink() if elm.is_file() else shutil.rmtree(elm)

    def __SAVE_CONFIGS(self):
        if not os.path.exists(self.cfgs.checkpoint_path):
            os.makedirs(self.cfgs.checkpoint_path)
        cfgs_dict = self.cfgs.__dict__
        filtered_cfgs_dict = {k: v for k, v in cfgs_dict.items() if k in self.REQUIRE_CONFGS}

        with open(self.cfgs.checkpoint_path + 'configs.yml', 'w') as yaml_file:
            yaml.dump(filtered_cfgs_dict, yaml_file, default_flow_style=False)

CFGS = Configs().cfgs
