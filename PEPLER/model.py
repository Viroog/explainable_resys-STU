import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class ContinuousPromptWithMF(nn.Module):
    def __init__(self, pretrained_model_name_or_path, user_nums, item_nums, token_nums, args):
        super(ContinuousPromptWithMF, self).__init__()
        self.user_nums = user_nums
        self.item_nums = item_nums
        self.args = args

        # 实例化gpt2LM模型，本质为Transformer+一个Linear进行语言生成
        self.gpt2 = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
        self.gpt2.resize_token_embeddings(token_nums)

        # 加入了bos, eos和空格，要重新定义gpt的word embedding大小

        # gpt2模型有两个Embedding: wpe(position) shape: (1024, 768) 1024个position 和 wte(token) shape: (50257, 768) 50257个单词
        # hidden_dimension
        self.d = self.gpt2.transformer.wte.weight.shape[1]

        self.user_embed = nn.Embedding(user_nums, self.d)
        self.item_embed = nn.Embedding(item_nums, self.d)

        nn.init.xavier_normal_(self.user_embed.weight.data)
        nn.init.xavier_normal_(self.item_embed.weight.data)

    # type: # tensor in cpu
    # 根据transformer文档，label设置为100的会被忽略掉
    def forward(self, user, item, seq, attention_mask, ignore_index=-100):

        # batch_size应该为传进来的数据的大小而不是args中的batch_size，否则会出现大小不匹配的情况
        batch_size = user.shape[0]

        # (batch_size, d)
        user_embed = self.user_embed(user.to(self.args.device))
        # (batch_size, d)
        item_embed = self.item_embed(item.to(self.args.device))
        # (batch_size, 24, d)
        word_embed = self.gpt2.transformer.wte(seq.to(self.args.device))

        # (batch_size, d) --> (batch_size, 1, d)
        # shape: (batch_size, 26, d)
        inputs_embed = torch.cat([user_embed.unsqueeze(1), item_embed.unsqueeze(1), word_embed], dim=1)

        # 先逐元素相乘，然后在最后一个维度上求和
        rating_pred = torch.sum(user_embed * item_embed, dim=-1)

        # mask, 在要在原本的mask基础上加上两个mask，即用户和id的mask
        # 原来的mask shape: (batch_size, seq_len + 4)
        # 多出的4个为两个空格+bos+eos
        user_item_mask = torch.ones((self.args.batch_size, 2), dtype=torch.int64)
        # 在第一维拼接
        total_mask = torch.cat([user_item_mask, attention_mask], dim=1)

        # label是one-hot vecotr每个输出对应一个vector，维度为token的大小 shape: (50259, )
        # total shape: (batch_size,  ,50259)

        # user & item对应的label, 产生由ignore_index填充的tensor
        # shape: (batch_size, 2)
        left_label = torch.full((batch_size, 2), ignore_index, dtype=torch.int64)
        # token对应的label
        # shape: (batch_size, 24)
        right_label = torch.where(attention_mask != 0, seq, torch.tensor(ignore_index, dtype=torch.int64))
        total_label = torch.cat([left_label, right_label], dim=1)

        text_pred = self.gpt2.forward(attention_mask=total_mask.to(self.args.device),
                                      inputs_embeds=inputs_embed.to(self.args.device),
                                      labels=total_label.to(self.args.device))

        return rating_pred, text_pred
