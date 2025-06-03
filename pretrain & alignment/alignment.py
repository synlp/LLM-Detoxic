#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm  # 导入 tqdm 用于进度条显示

# ---------------------------
# 模型定义：只有一个参数矩阵，将 embedding 从维度 embedding_dim1 映射到 embedding_dim2
# ---------------------------
class MappingModel(nn.Module):
    def __init__(self, embedding_dim1, embedding_dim2):
        super(MappingModel, self).__init__()
        # 随机初始化一个参数矩阵
        self.mapping = nn.Parameter(torch.randn(embedding_dim1, embedding_dim2))

    def forward(self, x):
        # x: [batch_size, embedding_dim1]
        # 返回: [batch_size, embedding_dim2]
        return torch.matmul(x, self.mapping)


# ---------------------------
# 数据集定义：每个样本对应公共词表中的一个词
# 对于每个词，返回：
#   - 来自大模型1的 embedding（作为正例输入）
#   - 来自大模型2的 embedding（作为正例目标）
#   - 随机采样的 n_negatives 个负例的大模型2 embedding（保证与正例不同）
# ---------------------------
class WordDataset(Dataset):
    def __init__(self, common_vocab, model1_emb, model2_emb, n_negatives):
        """
        :param common_vocab: list，公共词表中的每个词（字符串）
        :param model1_emb: dict，词 -> 大模型1的 embedding（tensor）
        :param model2_emb: dict，词 -> 大模型2的 embedding（tensor）
        :param n_negatives: int，每个正例采样的负例数量
        """
        self.common_vocab = common_vocab
        self.model1_emb = model1_emb
        self.model2_emb = model2_emb
        self.n_negatives = n_negatives
        self.vocab_size = len(common_vocab)

    def __len__(self):
        return len(self.common_vocab)

    def __getitem__(self, idx):
        word = self.common_vocab[idx]
        # 正例输入：大模型1中的 embedding
        pos_input = self.model1_emb[word]    # tensor, shape: [embedding_dim1]
        # 正例目标：大模型2中的 embedding
        pos_target = self.model2_emb[word]   # tensor, shape: [embedding_dim2]

        # 随机采样 n_negatives 个负例（避免采样到当前词）
        negatives = []
        while len(negatives) < self.n_negatives:
            neg_idx = random.randint(0, self.vocab_size - 1)
            neg_word = self.common_vocab[neg_idx]
            if neg_word == word:
                continue
            negatives.append(self.model2_emb[neg_word])
        negatives = torch.stack(negatives, dim=0)  # shape: [n_negatives, embedding_dim2]
        return pos_input, pos_target, negatives, word  # 最后返回的 word 仅供调试使用


# ---------------------------
# 通过 Hugging Face 加载预训练大模型及其 tokenizer，并提取输入 embedding 表征
# ---------------------------
def load_hf_embeddings(model_name):
    """
    加载 Hugging Face 模型及其 tokenizer，返回 token -> embedding 映射字典、tokenizer 对象以及 embedding 维度
    """
    print(f"加载 Hugging Face 模型：{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    # 获取模型的输入 embedding 层
    embed_layer = model.get_input_embeddings()
    embedding_weight = embed_layer.weight.data.cpu()  # shape: [vocab_size, embedding_dim]
    # 获取词表：tokenizer.get_vocab() 返回 {token: id}
    vocab_dict = tokenizer.get_vocab()
    # 构造 id -> token 的映射
    inv_vocab = {idx: token for token, idx in vocab_dict.items()}
    word2emb = {}
    for idx in range(embedding_weight.size(0)):
        token = inv_vocab.get(idx, None)
        if token is None:
            continue
        word2emb[token] = embedding_weight[idx]
    embedding_dim = embedding_weight.size(1)
    return word2emb, tokenizer, embedding_dim


# ---------------------------
# 主函数
# ---------------------------
def main(args):
    # 固定随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 单 GPU 训练（若有 GPU 则使用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # 加载大模型1和大模型2的 embedding 表征
    model1_emb, tokenizer1, embedding_dim1 = load_hf_embeddings(args.model1_name)
    model2_emb, tokenizer2, embedding_dim2 = load_hf_embeddings(args.model2_name)
    print(f"大模型1的 embedding 维度：{embedding_dim1}，大模型2的 embedding 维度：{embedding_dim2}")

    # 取两个模型词表的交集作为公共词表
    vocab1 = set(model1_emb.keys())
    vocab2 = set(model2_emb.keys())
    common_vocab = sorted(list(vocab1.intersection(vocab2)))
    if len(common_vocab) == 0:
        raise ValueError("两个模型的词表没有公共部分！")
    print(f"公共词表 V 中的词数：{len(common_vocab)}")

    # 构造数据集与 DataLoader
    dataset = WordDataset(common_vocab, model1_emb, model2_emb, args.n_negatives)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 初始化映射模型
    model = MappingModel(embedding_dim1, embedding_dim2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print("开始训练……")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        # 使用 tqdm 包装 dataloader 以显示进度条
        t = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
        for batch in t:
            # 每个 batch 包含：
            #   pos_input: [batch_size, embedding_dim1]
            #   pos_target: [batch_size, embedding_dim2]
            #   negatives: [batch_size, n_negatives, embedding_dim2]
            #   word: list[str]（调试用）
            pos_input, pos_target, negatives, _ = batch
            pos_input = pos_input.to(device).float()
            pos_target = pos_target.to(device).float()
            negatives = negatives.to(device).float()

            optimizer.zero_grad()
            # 映射后的输出 shape: [batch_size, embedding_dim2]
            mapped = model(pos_input)

            # 计算正例与负例的相似度（内积计算）
            sim_pos = torch.sum(mapped * pos_target, dim=1)           # shape: [batch_size]
            sim_neg = torch.sum(mapped.unsqueeze(1) * negatives, dim=2) # shape: [batch_size, n_negatives]
            # 拼接 logits，第 0 个位置为正例，其余为负例
            logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)  # shape: [batch_size, 1+n_negatives]

            # 标签均为 0，表示正例应当得分最高
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            # 更新 tqdm 进度条的描述信息
            t.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} 平均 Loss: {avg_loss:.4f}")

    # 保存训练后的模型参数
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "mapping_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"训练结束，模型已保存至 {save_path}")


# ---------------------------
# 程序入口，使用 argparse 解析命令行参数
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="基于 PyTorch 的映射模型训练代码：将 Hugging Face 大模型1 的 embedding 映射到大模型2 的 embedding 空间"
    )
    parser.add_argument("--model1_name", type=str, required=True,
                        help="Hugging Face 大模型1 的名称或路径（例如：bert-base-uncased）")
    parser.add_argument("--model2_name", type=str, required=True,
                        help="Hugging Face 大模型2 的名称或路径（例如：roberta-base）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="训练结束后保存映射模型的目录")
    parser.add_argument("--batch_size", type=int, default=32, help="训练时的 batch size")
    parser.add_argument("--epochs", type=int, default=10, help="训练的轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--n_negatives", type=int, default=10, help="每个正例采样的负例个数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    main(args)
