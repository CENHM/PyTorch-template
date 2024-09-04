import torch
import numpy as np
from torch.utils import data

import struct


class Dataset(data.Dataset):
    def __init__(self, train=True):
        # ------------- MODIFICATION FOR YOUR CODE ------------ #
        # ####################### START ####################### #

        self.train = train

        if self.train:
            self.x_path = './dataset/MNIST/train/train-images.idx3-ubyte'
            self.y_path = './dataset/MNIST/train/train-labels.idx1-ubyte'
        else:
            self.x_path = './dataset/MNIST/tests/t10k-images.idx3-ubyte'
            self.y_path = './dataset/MNIST/tests/t10k-labels.idx1-ubyte'

        # #######################  END  ####################### #
        
        self.x = self.__get_x()
        self.y = self.__get_y()


    def __get_x(self):
        # ------------- MODIFICATION FOR YOUR CODE ------------ #
        # ####################### START ####################### #

        if self.train:
            data_size = 47040016
        else:
            data_size = 7840016

        file_size = str(data_size - 16) + 'B'
        data_buf = open(self.x_path, 'rb').read()
        magic, num_images, num_row, num_column = struct.unpack_from('>IIII', data_buf, 0)
        datas = struct.unpack_from('>' + file_size, data_buf, struct.calcsize('>IIII'))
        datas = np.array(datas).reshape(num_images, 1, num_row, num_column)

        # #######################  END  ####################### #

        return datas
    

    def __get_y(self):
        # ------------- MODIFICATION FOR YOUR CODE ------------ #
        # ####################### START ####################### #

        if self.train:
            data_size = 60008
        else:
            data_size = 10008

        file_size = str(data_size - 8) + 'B'
        
        data_buf = open(self.y_path, 'rb').read()
        magic, num_labels = struct.unpack_from('>II', data_buf, 0)
        labels = struct.unpack_from('>' + file_size, data_buf, struct.calcsize('>II'))
        labels = np.array(labels).reshape(num_labels)

        # #######################  END  ####################### #
        return labels


    def __getitem__(self, idx):
        return torch.tensor(self.x[idx, :]).float(), torch.tensor(self.y[idx]).long()


    def __len__(self):
        return self.x.shape[0]