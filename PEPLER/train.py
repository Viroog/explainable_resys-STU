import argparse
from transformers import GPT2TokenizerFast
from data import DataLoader, Batchify

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
