import argparse
import os
import shutil
from pathlib import Path
import yaml


class Configs:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.init_configs()
        self.SKIP = ['testing', 'checkpoint_path', 'resume_test', 'checkpoint_file', 'result_path']
        self.cfgs = self.parser.parse_args()
        self.__PREPROCESS()


    def init_configs(self):
        # global setting
        self.parser.add_argument('--testing', default=False, action='store_true', help='Whether current running mode is train [default: True]')
        # training setting
        self.parser.add_argument('--checkpoint_path', default='tmp', help='Model checkpoint path [default: checkpoints\\tmp\\]')
        self.parser.add_argument('--checkpoint_file', type=str, default=None, help='Checkpoint .tar file for resuming')
        self.parser.add_argument('--epoch', type=int, default=5, help='Epoch to run [default: 5]')
        self.parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
        self.parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
        self.parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
        # testing setting
        self.parser.add_argument('--result_path', type=str, default='tmp', help='Save outputs path [default: results\\tmp\\]')


    def __PREPROCESS(self):
        self.__EDIT_CHECKPOINT_PATH()
        self.__EDIT_RESULT_PATH()
        if (not self.cfgs.testing) and (self.cfgs.checkpoint_file is not None) or (self.cfgs.testing):
            self.cfgs.resume_test = True
        else:
            self.cfgs.resume_test = False

        if not self.cfgs.resume_test:
            self.__CLEAR_CHECKPOINT_PATH_FILE()
            self.__SAVE_CONFIGS()
            return
        self.__LOAD_CHECKPOINT_CONFIGS()


    def __EDIT_CHECKPOINT_PATH(self):
        self.cfgs.checkpoint_path = 'checkpoints/' + self.cfgs.checkpoint_path + '/'


    def __EDIT_RESULT_PATH(self):
        self.cfgs.result_path = 'results/' + self.cfgs.result_path + '/'


    def __CLEAR_CHECKPOINT_PATH_FILE(self):
        for elm in Path(self.cfgs.checkpoint_path).glob('*'):
            elm.unlink() if elm.is_file() else shutil.rmtree(elm)


    def __SAVE_CONFIGS(self):
        if not os.path.exists(self.cfgs.checkpoint_path):
            os.makedirs(self.cfgs.checkpoint_path)
        cfgs_dict = self.cfgs.__dict__
        filtered_cfgs_dict = {k: v for k, v in cfgs_dict.items() if k not in self.SKIP}

        with open(self.cfgs.checkpoint_path + 'configs.yml', 'w') as yaml_file:
            yaml.dump(filtered_cfgs_dict, yaml_file, default_flow_style=False)


    def __LOAD_CHECKPOINT_CONFIGS(self):
        with open(self.cfgs.checkpoint_path + 'configs.yml', 'r') as yaml_file:
            loaded_cfgs = yaml.safe_load(yaml_file)

        for key, value in loaded_cfgs.items():
            if key in self.cfgs.__dict__:
                self.cfgs.__dict__[key] = value


cfgs = Configs().cfgs
