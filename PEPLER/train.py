import argparse

import torch.nn as nn
from transformers import GPT2TokenizerFast

from data import DataLoader, Batchify
from model import ContinuousPromptWithMF

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data/reviews.pickle',
                    help='the path of data used to train model')
# 每个文件包含了各个数据对应的索引
parser.add_argument('--index_dir', type=str, default='./data/1/')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='the max training epoch')
parser.add_argument('--batch_size', type=int, default=128, help='the number of data in one batch')
parser.add_argument('--seq_len', type=int, default=20, help='the number of words want to generate')
parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')

parser.add_argument('--pretrained_model_name_or_path', type=str, default='gpt2',
                    help='the pre-trained model want to use, like: gpt2/bert...')
parser.add_argument('--rating_reg', type=float, default=0.01, help='the parameter lambda in paper')

args = parser.parse_args()

print('Loading data...')
# begin of sentence
bos = '<bos>'
# end of sentence
eos = '<eos>'
# padding
pad = 'pad'
# GPT2用于将输入转换成Token的tokenizer, word->id
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad)
# # for tokenizer example
# text = 'I like recommender system'
# tokens = tokenizer.encode(text)
# print(tokens)
# print(tokenizer.decode(tokens))

corpus = DataLoader(args.data_path, args.index_dir, tokenizer, args.seq_len)
train_data = Batchify(corpus.train_data, tokenizer, bos, eos, args.batch_size, shuffle=True)
valid_data = Batchify(corpus.valid_data, tokenizer, bos, eos, args.batch_size, shuffle=False)
test_data = Batchify(corpus.test_data, tokenizer, bos, eos, args.batch_size, shuffle=False)

user_nums = len(corpus.user_dict)
item_nums = len(corpus.item_dict)
token_nums = len(tokenizer)

model = ContinuousPromptWithMF(args.pretrained_model_name_or_path, user_nums, item_nums, token_nums, args).to(args.device)

rating_criterion = nn.MSELoss()

model.train()
for epoch in range(args.epochs):
    user, item, rating, seq, attention_mask = train_data.next_batch()

    rating_pred, text_pred = model(user, item, seq, attention_mask)

    rating_loss = rating_criterion(rating_pred, rating)
    text_loss = text_pred.loss

    total_loss = text_loss + args.rating_reg * rating_loss