import torch
from torch.utils.data import Dataset
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('./consts.py'))))
from KPRN import consts


class TrainInteractionData(Dataset):
    def __init__(self, train_path_file):
        self.file = 'data/' + consts.PATH_DATA_DIR + '/' + train_path_file
        self.interaction_nums = 0
        self.interactions = []

        with open(self.file, 'r') as file:
            for line in file.readlines():
                self.interactions.append(eval(line.strip()))

        self.interaction_nums = len(self.interactions)

    def __getitem__(self, idx):
        return self.interactions[idx]

    def __len__(self):
        return self.interaction_nums


class TestInteractionData(Dataset):
    def __init__(self, test_interaction):
        self.data = test_interaction

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


# dataloader按照batch进行读取数据时，是取出大小为batch size的index列表，然后将index列表输入到dataset中(即TrainInteractionData类)的getitem方法中
# 取出对应index的数据，然后进行堆叠，形成一个batch的数据
# 因为我们的数据每一行是一个元组，(paths, label)，客制化定义my_collate函数
def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)

    return [data, target]


def sort_batch(batch, idxs, lengths):
    seq_lengths, permute_idx = lengths.sort(descending=True)
    seq_batch, seq_idxs = batch[permute_idx], idxs[permute_idx]

    return seq_batch, seq_idxs, seq_lengths
