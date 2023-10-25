import argparse
import os.path

import torch
import torch.nn as nn
from transformers import GPT2TokenizerFast

from data import DataLoader, Batchify
from model import ContinuousPromptWithMF
import torch.optim as optim

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

parser.add_argument('--mode', type=str, default='predict', help='choose mode: train or predict')

args = parser.parse_args()
model_path = f"pepler_lr={args.lr}_seqlen={args.seq_len}.pth"

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

model = ContinuousPromptWithMF(args.pretrained_model_name_or_path, user_nums, item_nums, token_nums, args).to(
    args.device)

rating_criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr)


def train():
    model.train()
    t_ra_loss, t_te_loss = 0, 0
    while True:
        # tensor in cpu
        user, item, rating, seq, attention_mask = train_data.next_batch()

        rating_pred, text_pred = model(user, item, seq, attention_mask)

        rating_loss = rating_criterion(rating_pred, rating.to(args.device))
        text_loss = text_pred.loss

        total_loss = text_loss + args.rating_reg * rating_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        t_ra_loss += rating_loss.item()
        t_te_loss += text_loss.item()

        if train_data.step == train_data.total_step:
            break

    return t_ra_loss / train_data.sample_num, t_te_loss / train_data.sample_num


# for valid_data or test data
def evaluation(data):
    model.eval()
    t_loss = 0
    while True:
        user, item, rating, seq, attention_mask = data.next_batch()

        rating_pred, text_pred = model(user, item, seq, attention_mask)

        rating_loss = rating_criterion(rating_pred, rating.to(args.device))
        text_loss = text_pred.loss

        total_loss = text_loss + args.rating_reg * rating_loss
        t_loss += total_loss.item()

        if data.step == data.total_step:
            break

    return t_loss / data.sample_num


# 测试的时候要产生文本，而不是进行分数预测
def test():
    model.eval()
    # 预测最终产生的结果
    total_rating, total_text = [], []
    while True:
        user, item, rating, seq, attention_mask = test_data.next_batch()

        # 起始text只有一个bos，即输入为user, item, bos
        # 然后每次产生一个新的输出再和前面的输入合并，形成新的输入
        # shape: (batch_size, 1) <bos>，不能是seq[:, 0]，要不然形状变了
        text = seq[:, :1]

        # 期望产生和seq一样长的输出
        for i in range(seq.shape[1]):
            # 只有第一步需要预测评分，后面的每一步都是产生一个新单词
            if i == 0:
                # shape: (batch_size, 1), shape: (batch_size, word_size)
                rating_pred, text_pred = model(user, item, text, None)
                # tolist()是让第二维的元素形成一个list
                total_rating.extend(rating_pred.tolist())
            else:
                _, text_pred = model(user, item, text, None, False)

            # shape: (batch_size, seq_len, word_size) before softmax
            logits = text_pred.logits
            # 只要最后一个输出，即最新生成的token
            # shape: (batch_size, word_size)
            last_token = logits[:, -1, :]
            word_prob = torch.softmax(last_token, dim=-1)
            # 选择概率最大的那个，并keepdim
            # shape: (batch_size, 1)
            word = torch.argmax(word_prob, dim=-1, keepdim=True)
            # 将最新产生单词放入text中
            # text in cpu but word in cuda
            text = torch.cat([text, word.to('cpu')], dim=-1)

        # 移除第0个，即<bos>
        total_text.extend(text[:, 1:].tolist())

        if test_data.step == test_data.total_step:
            break

    return total_rating, total_text


# 需要训练，或者不需要训练但是模型的保存文件不存在
if args.mode == 'train' or (args.mode == 'predict' and os.path.exists(model_path) is False):
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # rating loss, text loss
        loss1, loss2 = train()
        print(f"epoch: {epoch}, rating loss: {loss1}, text loss: {loss2}")

        # 每10个epoch在验证集上验证一次，并保存loss最小的那个模型
        if (epoch + 1) % 10 == 0:
            val_loss = evaluation(valid_data)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model, model_path)
# 直接加载模型进行预测
elif args.mode == 'predict':
    model = torch.load(model_path).to(args.device)

# test code
# 验证集和训练集是没有打乱的，因此在预测过程中可以不存储ground_truth
test_loss = evaluation(test_data)
print(f"for test loss: {test_loss}")
rating_pred, text_pred = test()

