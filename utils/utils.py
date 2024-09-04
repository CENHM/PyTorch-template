import random
import numpy as np
import torch
import os
import datetime

class Logger:
    def __init__(self, testing, checkpoint_path, checkpoint_file, result_path):
        self.testing = testing
        self.path = 'log.txt'
        self.checkpoint_path = checkpoint_path
        self.result_path = result_path

        self.save_path = self.checkpoint_path if not self.testing else self.result_path
        self.file_name = 'log.txt' if not self.testing else f'{checkpoint_path.split("/")[-2]}-{checkpoint_file}.txt'

        self.CLEAR_LOG()
        

    def CLEAR_LOG(self):
        with open(self.path, 'w+') as f:
            pass


    def SAVE_LOG(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        with open(self.path, 'r') as original:
            with open(self.save_path + self.file_name, 'a') as copy:
                copy.write(f'\n********* {datetime.datetime.now()} *********\n\n')
                for line in original:
                    copy.write(line)
        
        self.CLEAR_LOG()


    def WRITE(self, text):
        with open(self.path, 'a+') as f:
            f.write(text + '\n')


def SET_SEED(seed=1024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def LOAD_CHECKPOINT(optimizer, model, path, file):
    checkpoint_path = path + '/' + file + '.tar'
    checkpoint = torch.load(checkpoint_path)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    return start_epoch, optimizer, model
    

def LOAD_WEIGHT(model, path, file):
    checkpoint_path = path + '/' + file + '.tar'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def SAVE_CHECKPOINT(epoch, optimizer, model, path):
    save_dict = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict()
    }
    torch.save(save_dict, os.path.join(path, f'EPOCH-{epoch}.tar'))


def VISUALIZE_TENSOR_SHAPE():
    normal_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"

