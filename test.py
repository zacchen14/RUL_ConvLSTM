import numpy as np

from convlstm import EncoderDecoder
import torch
import numpy as np
import pandas as pd
import os
import torch.utils.data as Data
from matplotlib import pyplot as plt
import torch.nn.functional as F

data_x = np.load('/home/zacchen14/PycharmProjects/ConvLSTM_pytorch/data/CMAPSSData/16_8/train_x_FD002.npy')

print(data_x.shape)