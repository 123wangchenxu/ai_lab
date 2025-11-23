# transformer_with_both_comments.py
# 可运行的 Transformer 修正版：同时保留“原始注释”和“修正说明”（中文）
import math
import random
import time
import os
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------
# Embeddings（词嵌入）
# ---------------------------

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model, pad_idx=0):
        super().__init__()
        # 初始编码(w*d的那个词表)
        # params: vocab_size 词汇表有多少个 d_model:缩放维度
        # 创建初始W*d矩阵(encoder)
        # 修正说明：使用 padding_idx 保证 PAD 的 embedding 恒为 0 并且不被更新（避免 pad 被学坏）
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.dmodel = d_model

    def forward(self, x):
        # 初始归一化
        return self.embed(x) * math.sqrt(self.dmodel)


# ---------------------------
# PositionEmbedding（位置编码）
# ---------------------------

class PositionEmbedding(nn.Module):
    # 位置编码
    # 创建一个位置向量pe，只需要生成max_len长度，那个公式d_model都是固定的，只有pos不固定，
    # 每一次都是去截取，有最长的，截一部分用就行了
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # 修正：arange
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    # 这里就是输入过来的，分批次往上加，比如对于一个batch，里面句子长度3,4,5，统一扩到5，加，最后把多的掩码掉
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


# ---------------------------
# attention（缩放点积注意力）
# ---------------------------

# attention
# 值得注意的点是加了dropout，其余就是公式
def attention(query, key, value, mask: Optional[torch.Tensor] = None, dropout: Optional[nn.Dropout] = None):
    """
    query/key/value: [batch, head, seq, d_k]
    mask: 布尔张量或 None。True 表示 keep，False 表示要屏蔽。
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 缩放点积
    # 这里mask取反，对于mask为0的位置值变成true，意思是需要掩盖，掩盖值为inf
    if mask is not None:
        # mask True 表示保留。用 ~mask 掩盖（置 -inf）
        scores = scores.masked_fill(~mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    output = torch.matmul(attn, value)
    return output, attn


# ---------------------------
# MultiHeadAttention（多头注意力）
# ---------------------------

# 这里线性层的作用是生成d*d的矩阵,加偏置是固定做法，w*d @ d*d = w*d
# 多头的含义是多个线性空间，实际上是把d变成head*d_head,最后拼回来，out线性层用于多个子空间拼回来的融合
class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0, "d_model 必须能被 head 数整除"
        # 这里加断言是因为d_model被拆成h*d_k了,必须能拆才多头
        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h

        # self.linear_q/k/v 和 self.linear_out
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    # 这里采用的方法是先把h和l互换，变成h个L，d_head的矩阵去做attention，然后做好之后就是h个L,d_head
    # 在把L和h换过来，h*d_head拼成一维
    def _split_heads(self, x):
        # x: [batch, seq, d_model] -> [batch, head, seq, d_k]
        batch = x.size(0)
        return x.view(batch, -1, self.h, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        # x: [batch, head, seq, d_k] -> [batch, seq, d_model]
        batch = x.size(0)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        return x

    def forward(self, query, key, value, mask: Optional[torch.Tensor] = None):
        """
        query/key/value: [batch, seq, d_model]
        mask: [batch, seq_q, seq_k] 或可带 head 维度
        """
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        # 这里mask是关注之前的padding
        if mask is not None:
            # 若 mask 是 [batch, seq_q, seq_k]，增加 head 维度以便广播
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # -> [batch, 1, seq_q, seq_k]
        x, attn = attention(q, k, v, mask=mask, dropout=self.dropout)
        x = self._merge_heads(x)
        return self.linear_out(x)


# ---------------------------
# FeedForward（前馈网络）
# ---------------------------

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


# ---------------------------
# AddNorm（残差 + LayerNorm）
# ---------------------------

# 这里是先定义两个层，x是残差连接，先进行层归一化，可以让训练更稳定，和原论文有出入
class AddNorm(nn.Module):
    def __init__(self, size, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 原始注释：return x + self.dropout(sublayer(self.norm(x)))
        # 修正说明：sublayer_fn 是 callable，先做 LayerNorm 再把规范化结果传入子层
        return x + self.dropout(sublayer(self.norm(x)))


# ---------------------------
# EncoderLayer
# ---------------------------

# 原始注释：
# # 这里module_list的作用是保存参数
class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn: MultiHeadAttention, feed_forward: FeedForward, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 原始注释：self.subLayers = nn.ModuleList([...])
        # 修正说明：统一命名为 self.sublayers，避免大小写不一致导致 AttributeError
        self.sublayers = nn.ModuleList([
            AddNorm(d_model, dropout),
            AddNorm(d_model, dropout)
        ])

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # 原始注释：x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 修正说明：使用 pre-norm（先 LayerNorm 再子层），更稳定
        x = self.sublayers[0](x, lambda x_norm: self.self_attn(x_norm, x_norm, x_norm, mask))
        x = self.sublayers[1](x, lambda x_norm: self.feed_forward(x_norm))
        return x


# ---------------------------
# DecoderLayer（解码器层）
# ---------------------------

# 原始注释：
# # 这里和encoder区别在于多了一个交叉注意力，用的是memory，也就是来源于encoder的初始qk，这里初始都是一样的
# # 此外tag_mask表示的是掩码
class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn: MultiHeadAttention, cross_attn: MultiHeadAttention,
                 feed_forward: FeedForward, dropout=0.1):
        super().__init__()
        # 保留原始命名 self.attn / self.cross_attn
        self.attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        # 原始注释：self.subLayers = nn.ModuleList([...])
        # 修正说明：统一为 self.sublayers
        self.sublayers = nn.ModuleList([
            AddNorm(d_model, dropout),
            AddNorm(d_model, dropout),
            AddNorm(d_model, dropout)
        ])

    def forward(self, x, memory, src_mask: Optional[torch.Tensor], tag_mask: Optional[torch.Tensor]):
        # 原始注释：
        # x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tag_mask))
        # x = self.sublayers[1](x, lambda x: self.cross_attn(x, memory, memory, src_mask))
        # x = self.sublayers[2](x, lambda x: self.feed_forward(x))
        # 修正说明：按 pre-norm 顺序调用 masked self-attn -> cross-attn -> feed-forward
        x = self.sublayers[0](x, lambda x_norm: self.attn(x_norm, x_norm, x_norm, tag_mask))
        x = self.sublayers[1](x, lambda x_norm: self.cross_attn(x_norm, memory, memory, src_mask))
        x = self.sublayers[2](x, lambda x_norm: self.feed_forward(x_norm))
        return x


# ---------------------------
# Transformer（整体模型）
# ---------------------------

# 原始注释：
# # 首先src_embed,和tag_embed都是编码
# # attn、fn定义注意力函数和前馈
# # encoder在这个基础上*N
# # 定义输出层，让每一个转化的标签都有对应值
class Transformer(nn.Module):
    def __init__(self, src_vocab, tag_vocab, d_model=512, N=6, h=8, d_ff=2048, dropout=0.1, pad_idx=0):
        super().__init__()
        # 保留原始命名 src_embed / tag_embed
        self.src_embed = nn.Sequential(
            Embeddings(src_vocab, d_model, pad_idx=pad_idx),
            PositionEmbedding(d_model)
        )
        self.tag_embed = nn.Sequential(
            Embeddings(tag_vocab, d_model, pad_idx=pad_idx),
            PositionEmbedding(d_model)
        )

        # 原始注释：attn = lambda: MultiHeadAttention(h, d_model, dropout)
        # 修正说明：使用工厂函数方便每层生成独立参数
        def attn(): return MultiHeadAttention(h, d_model, dropout)

        def ff(): return FeedForward(d_model, d_ff, dropout)

        self.encoder = nn.ModuleList([EncoderLayer(d_model, attn(), ff(), dropout) for _ in range(N)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, attn(), attn(), ff(), dropout) for _ in range(N)])
        self.out = nn.Linear(d_model, tag_vocab)
        # 记录 pad idx 便于 mask 与 loss 使用
        self.pad_idx = pad_idx
        self.d_model = d_model

    def encode(self, src, src_mask: Optional[torch.Tensor]):
        # 原始注释：def encode(self, src, src_mask): x = self.src_embed(src) for layer in self.encoder: x = layer(x, src_mask) return x
        x = self.src_embed(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tag, memory, tag_mask: Optional[torch.Tensor], src_mask: Optional[torch.Tensor]):
        # 原始注释：def decode(self, tag, memory, tag_mask):
        #                 x = self.tag_embed(tag)
        #                 for layer in self.decoder:
        #                     x = layer(x, memory, tag_mask)
        #                 return x
        x = self.tag_embed(tag)
        for layer in self.decoder:
            x = layer(x, memory, src_mask, tag_mask)
        return x

    def forward(self, src, tag, src_mask: Optional[torch.Tensor] = None, tag_mask: Optional[torch.Tensor] = None):
        # 原始注释：def forward(self, src, tag, src_mask=None, tag_mask=None):
        #                 memory = self.encode(src, src_mask)
        #                 out = self.decode(tag, memory, tag_mask)
        #                 return self.out(out)
        memory = self.encode(src, src_mask)
        # 注意：decode 调用这里和原始签名顺序保持一致（tag_mask, src_mask 在内部会传给子层）
        out = self.decode(tag, memory, tag_mask, src_mask)
        return self.out(out)


# ---------------------------
# Mask helpers（掩码工具）
# ---------------------------

# 原始代码中没有将 mask 处理细化，这里补上并注释
def make_src_mask(src: torch.Tensor, pad_idx: int = 0):
    """
    原始注释（隐含）: src_mask 用于屏蔽 encoder 中的 padding
    修正说明：返回布尔 mask，形状为 [batch, 1, 1, src_seq]，True 表示可 attend
    """
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_tag_mask(tag: torch.Tensor, pad_idx: int = 0):
    """
    原始注释（隐含）: tag_mask 用于 decoder 自注意力的 padding + subsequent 屏蔽
    修正说明：返回布尔 mask，形状 [batch,1,tgt_len,tgt_len]
    """
    batch, tgt_len = tag.size()
    pad_mask = (tag != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch,1,1,tgt_len]
    subsequent = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=tag.device)).unsqueeze(0).unsqueeze(
        1)
    return pad_mask & subsequent


# ---------------------------
# Toy dataset（示例数据集，可替换）
# ---------------------------

class ToySeqDataset(Dataset):
    """
    原始注释（类似）：用于生成随机 src/tag 对作为演示
    修正说明：你可以把这里替换为真实数据加载逻辑（只需返回已 pad 的 LongTensor）
    """

    def __init__(self, num_samples, src_len, tag_len, src_vocab, tag_vocab, pad_idx=0):
        super().__init__()
        self.samples = []
        for _ in range(num_samples):
            r_src = random.randint(1, src_len)
            r_tag = random.randint(2, tag_len)
            # 使用 1 作为 start token（示例）
            src = [random.randint(2, src_vocab - 1) for _ in range(r_src)]
            tag = [1] + [random.randint(2, tag_vocab - 1) for _ in range(r_tag - 1)]
            src = src + [pad_idx] * (src_len - len(src))
            tag = tag + [pad_idx] * (tag_len - len(tag))
            self.samples.append((torch.LongTensor(src), torch.LongTensor(tag)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    srcs = torch.stack([b[0] for b in batch], dim=0)
    tags = torch.stack([b[1] for b in batch], dim=0)
    return srcs, tags


# ---------------------------
# 训练 / 验证 / 贪婪解码（推理）
# ---------------------------

def train_epoch(model: Transformer, dataloader: DataLoader, optimizer, criterion, device, clip=None):
    model.train()
    total_loss = 0.0
    for src, tag in dataloader:
        src = src.to(device)
        tag = tag.to(device)
        # 原始注释（常见）：decoder 输入是 tag[:, :-1]，目标是 tag[:, 1:]
        tag_input = tag[:, :-1]
        tag_target = tag[:, 1:]

        src_mask = make_src_mask(src, model.pad_idx).to(device)
        tag_mask = make_tag_mask(tag_input, model.pad_idx).to(device)

        logits = model(src, tag_input, src_mask=src_mask, tag_mask=tag_mask)  # [batch, tgt_seq-1, vocab]
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = tag_target.contiguous().view(-1)

        loss = criterion(logits_flat, target_flat)
        optimizer.zero_grad()
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item() * src.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(model: Transformer, dataloader: DataLoader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for src, tag in dataloader:
            src = src.to(device)
            tag = tag.to(device)
            tag_input = tag[:, :-1]
            tag_target = tag[:, 1:]

            src_mask = make_src_mask(src, model.pad_idx).to(device)
            tag_mask = make_tag_mask(tag_input, model.pad_idx).to(device)

            logits = model(src, tag_input, src_mask=src_mask, tag_mask=tag_mask)
            logits_flat = logits.view(-1, logits.size(-1))
            target_flat = tag_target.contiguous().view(-1)

            loss = criterion(logits_flat, target_flat)
            total_loss += loss.item() * src.size(0)

            preds = logits.argmax(dim=-1)  # [batch, tgt_seq-1]
            mask = (tag_target != model.pad_idx)
            total_correct += (preds == tag_target).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, acc


def greedy_decode(model: Transformer, src: torch.Tensor, max_len: int, device, start_symbol: int = 1):
    """
    原始注释（类似）：简单的自回归贪婪解码
    修正说明：逐步把已生成序列 ys 送入 decoder，每步取 logits 的 argmax
    """
    model.eval()
    src = src.to(device)
    src_mask = make_src_mask(src, model.pad_idx).to(device)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1, dtype=torch.long, device=device) * start_symbol
    with torch.no_grad():
        for i in range(max_len - 1):
            tag_mask = make_tag_mask(ys, model.pad_idx).to(device)
            # 注意 decode 函数参数顺序：decode(tag, memory, tag_mask, src_mask)
            out = model.decode(ys, memory, tag_mask, src_mask)
            logits = model.out(out[:, -1:, :])  # 取最后一步的 logits
            prob = F.softmax(logits.squeeze(1), dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            next_word = next_word.unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)
    return ys


# ---------------------------
# main: 示例训练流程（可直接运行）
# ---------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 超参（示例）
    SRC_VOCAB = 500
    TAG_VOCAB = 500
    PAD_IDX = 0
    START_IDX = 1
    SRC_LEN = 30
    TAG_LEN = 30
    BATCH = 32
    TRAIN_SAMPLES = 2000
    VAL_SAMPLES = 400
    EPOCHS = 6
    D_MODEL = 128
    N_LAYERS = 4
    H = 8
    D_FF = 512
    LR = 1e-3
    CLIP = 1.0
    SAVE_PATH = "transformer_best_with_comments.pt"

    # 固定随机种子（可选）
    random.seed(42)
    torch.manual_seed(42)

    model = Transformer(SRC_VOCAB, TAG_VOCAB, d_model=D_MODEL, N=N_LAYERS, h=H, d_ff=D_FF, dropout=0.1, pad_idx=PAD_IDX)
    model = model.to(device)

    train_ds = ToySeqDataset(TRAIN_SAMPLES, SRC_LEN, TAG_LEN, SRC_VOCAB, TAG_VOCAB, pad_idx=PAD_IDX)
    val_ds = ToySeqDataset(VAL_SAMPLES, SRC_LEN, TAG_LEN, SRC_VOCAB, TAG_VOCAB, pad_idx=PAD_IDX)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, collate_fn=collate_fn)

    criterion = nn.CrossEnt
