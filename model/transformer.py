import torch
from torch import nn
import torch.nn.functional as F
import math


# 初始编码(w*d的那个词表)
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        # params: vocab_size 词汇表有多少个 d_model:缩放维度
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)  # 创建初始W*d矩阵(encoder)
        self.dmodel = d_model

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.dmodel)  # 初始归一化


# 位置编码
class PositionEmbedding(nn.Module):
    # 创建一个位置向量pe，只需要生成max_len长度，那个公式d_model都是固定的，只有pos不固定，
    # 每一次都是去截取，有最长的，截一部分用就行了
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arrange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arrange(0, d_model, 2).float() * -math.log(10000) / d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    # 这里就是输入过来的，分批次往上加，比如对于一个batch，里面句子长度3,4,5，统一扩到5，加，最后把多的掩码掉
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# attention
# 值得注意的点是加了dropout，其余就是公式
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
    if scores is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return attn @ value, attn


# 这里线性层的作用是生成d*d的矩阵,加偏置是固定做法，w*d @ d*d = w*d
# 多头的含义是多个线性空间，实际上是把d变成head*d_head,最后拼回来，out线性层用于多个子空间拼回来的融合
class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    # 这里采用的方法是先把h和l互换，变成h个L，d_head的矩阵去做attention，然后做好之后就是h个L,d_head
    # 在把L和h换过来，h*d_head拼成一维
    def forward(self, query, key, value, masked=None):
        batch_size = query.size(0)

        def transform(x, linear):
            x = linear(x)
            return x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        query = transform(query, self.linear_q)
        key = transform(key, self.linear_k)
        value = transform(value, self.linear_v)
        x, _ = attention(query, key, value, mask=masked, dropout=self.dropout)
        x = x.transpose(1, 2).contigous.view(batch_size, -1, self.d_model)
        return self.linear_out(x)


# 这里就是先升维度，后relu(非线性，比线性更好),后降维，便于残差连接，最后dropout防止过拟合
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# 这里是先定义两个层，x是残差连接，先进行层归一化，可以让训练更稳定，和原论文有出入
class AddNorm(nn.Module):
    def __init__(self, size, dropout=0.1):
        super().__init__()
        self.normal = nn.LayerNorm
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# 这里module_list的作用是保存参数

class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.subLayers = nn.ModuleList(
            [
                AddNorm(d_model, dropout),
                AddNorm(d_model, dropout)
            ]
        )

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayers[1](x, lambda x: self.feed_forward(x))
        return x


# 这里和encoder区别在于多了一个交叉注意力，用的是memory，也就是来源于encoder的初始qk，这里初始都是一样的
# 此外tag_mask表示的是掩码
class DecoderLayer(nn.Module):
    def __init__(self, d_model, cross_attn, self_attn, feed_forward, dropout):
        super().__init__()
        self.attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.subLayers = nn.ModuleList(
            [
                AddNorm(d_model, dropout),
                AddNorm(d_model, dropout),
                AddNorm(d_model, dropout),
            ]
        )

    def forward(self, x, memory, src_mask, tag_mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tag_mask))
        x = self.sublayers[1](x, lambda x: self.cross_attn(x, memory, memory, src_mask))
        x = self.sublayers[2](x, lambda x: self.feed_forward(x))
        return x


# 首先src_embed,和tag_embed都是编码
# attn、fn定义注意力函数和前馈
# encoder在这个基础上*N
# 定义输出层，让每一个转化的标签都有对应值

class Transformer(nn.Module):
    def __init__(self, src_vocab, tag_vocab, d_model=512, N=6, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.src_embed = nn.Sequential(
            Embeddings(src_vocab, d_model),
            PositionEmbedding(d_model)
        )
        self.tag_embed = nn.Sequential(
            Embeddings(tag_vocab, d_model),
            PositionEmbedding(d_model)
        )
        attn = lambda: MultiHeadAttention(h, d_model, dropout)
        ff = lambda: FeedForward(d_model, d_ff, dropout)
        self.encoder = nn.ModuleList(
            [
                EncoderLayer(d_model, attn(), ff()) for _ in range(N)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(d_model, attn(), attn(), ff(), dropout) for _ in range(N)
            ]
        )
        self.out = nn.Linear(d_model, tag_vocab)

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tag, memory, tag_mask):
        x = self.tag_embed(tag)
        for layer in self.decoder:
            x = layer(x, memory, tag_mask)
        return x

    def forward(self, src, tag, src_mask=None, tag_mask=None):
        memory = self.encode(src, src_mask)
        out = self.decode(tag, memory, tag_mask)
        return self.out(out)


# 调用示例
model = Transformer(src_vocab=1000, tag_vocab=500, d_model=512, N=6, h=8, d_ff=2048)
