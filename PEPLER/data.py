import math
import pickle
import random

import torch


# 相当于在存储数据的时候就进行了从0开始的映射
class MappingDict:
    def __init__(self):
        self.mapping_dict = {}

    def add(self, elem):
        if elem not in self.mapping_dict.keys():
            self.mapping_dict[elem] = len(self.mapping_dict)

    # 特殊方法，使用len()能直接获取长度
    def __len__(self):
        return len(self.mapping_dict)


class DataLoader:
    def __init__(self, data_path, index_dir, tokenizer, seq_len):
        self.data_path = data_path
        self.index_dir = index_dir
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.user_dict, self.item_dict = MappingDict(), MappingDict()
        # 评分的最大值和最小值，如果预测的分数在这个区间之外，则直接等于最大/小值
        self.max_rating, self.min_rating = float('-inf'), float('inf')
        self.initialize()

        # 这个feature在计算指标会用到，训练模型的时候没有用到
        self.feature_set = set()
        # 获取train, valid和test(与源码不一致，这里只实现continuous prompt，不需要将物品和用户映射到特征)
        self.train_data, self.valid_data, self.test_data = self.load_data()

    def initialize(self):

        with open(self.data_path, 'rb') as f:
            reviews = pickle.load(f)

            # 每个review是一个dict()
            for review in reviews:
                self.user_dict.add(review['user'])
                self.item_dict.add(review['item'])

                rating = review['rating']

                self.max_rating = max(self.max_rating, rating)
                self.min_rating = min(self.min_rating, rating)

    def load_idx(self):
        # train
        with open(self.index_dir + 'train.index', 'r') as f:
            # 将整个文件读取成一行字符串
            train_idx = [int(x) for x in f.readline().split(' ')]
        # valid
        with open(self.index_dir + 'validation.index', 'r') as f:
            valid_idx = [int(x) for x in f.readline().split(' ')]
        # test
        with open(self.index_dir + 'test.index', 'r') as f:
            test_idx = [int(x) for x in f.readline().split(' ')]

        return train_idx, valid_idx, test_idx

    def load_data(self):
        data = []

        with open(self.data_path, 'rb') as f:
            reviews = pickle.load(f)

            for review in reviews:
                (feature, _, u_review, _) = review['template']
                tokens = self.tokenizer.encode(u_review)
                # 取前seq_len个
                text = self.tokenizer.decode(tokens[:self.seq_len])

                user, item, rating = review['user'], review['item'], review['rating']
                data.append({
                    'user': self.user_dict.mapping_dict[user],
                    'item': self.item_dict.mapping_dict[item],
                    'rating': rating,
                    'text': text,
                    'feature': feature
                })

                self.feature_set.add(feature)

        train_idx, valid_idx, test_idx = self.load_idx()
        train_data, valid_data, test_data = [], [], []

        for idx in train_idx:
            train_data.append(data[idx])
        for idx in valid_idx:
            valid_data.append(data[idx])
        for idx in test_idx:
            test_data.append(data[idx])

        return train_data, valid_data, test_data


# 用于获得一个batch的数据
class Batchify:
    def __init__(self, data, tokenizer, bos, eos, batch_size=128, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle

        user, item, rating, text = [], [], [], []
        for x in data:
            user.append(x['user'])
            item.append(x['item'])
            rating.append(x['rating'])
            # 将bos和eos加入到句子中
            text.append(f'{bos} {x["text"]} {eos}')

        # 将评论一起进行tokenizer，并进行填充，并以pytorh的张量返回
        encoded_inputs = tokenizer(text, padding=True, return_tensors='pt')
        # contiguous()保证张量是连续存储的
        # self.seq = encoded_inputs['input_ids'].contiguous()
        # # 这里的attention_mask是用来mask在tokenize过程中padding的位置，保证在计算attention score时被忽略掉
        # self.attention_mask = encoded_inputs['attention_mask'].contiguous()
        # self.user = torch.LongTensor(user).contiguous()
        # self.item = torch.LongTensor(item).contiguous()
        # self.rating = torch.LongTensor(rating).contiguous()

        self.seq = encoded_inputs['input_ids']
        self.attention_mask = encoded_inputs['attention_mask']
        self.user = torch.LongTensor(user)
        self.item = torch.LongTensor(item)
        self.rating = torch.LongTensor(rating)

        # 总的数据条数
        self.sample_num = len(data)
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.index_list = list(range(self.sample_num))
        self.step = 0

    def next_batch(self):
        # 全部数据都过了一遍，重新开始
        if self.step == self.total_step:
            self.step = 0
            # shuffle
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        end = min(start + self.batch_size, self.sample_num)
        self.step += 1

        idx = self.index_list[start:end]
        # shape: (batch_size, )
        user, item, rating = self.user[idx], self.item[idx], self.rating[idx]
        # shape: (batch_size, seq_len)
        seq, attention_mask = self.seq[idx], self.attention_mask[idx]

        return user, item, rating, seq, attention_mask
