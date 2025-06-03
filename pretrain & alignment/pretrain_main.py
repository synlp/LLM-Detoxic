from model import LlamaForCausalLMTeacher
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, Trainer
from typing import Optional
import csv
import random

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    training_data_path: str = field(
        default=None,
        metadata={"help": "Path to the training CSV data file."}
    )

class MyCollator:
    def __init__(self, tokenizer, pad_token_id, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.label_pad_token_id
        )
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 每个样本为一行 CSV，第一列为输入文本，第二列为输出文本
        example = self.data[idx]
        input_text = example["vanilla adversarial"].strip()
        target_text = example["completion"].strip()

        # 这里构造了训练时的完整文本：输入 + EOS + 输出 + EOS
        full_text = input_text + '\n' + target_text + self.tokenizer.eos_token
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # 对输入部分进行 mask（不计算损失），只让模型在输出部分计算损失
        # 这里先单独对输入文本进行编码，计算其 token 数量（注意：不自动添加特殊 token）
        prompt_ids = self.tokenizer(input_text + '\n', add_special_tokens=False)['input_ids']
        # 加上 EOS 分隔符
        prompt_length = len(prompt_ids)

        labels = input_ids.copy()
        for i in range(prompt_length):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels)
        }

# 解析参数（包括模型、数据、训练参数）
parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path, use_fast=False, trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLMTeacher.from_pretrained(
    model_args.model_name_or_path, trust_remote_code=True
)


# model.reset_para()
# print('parameter re-initialized')

# 读取 CSV 数据：只保留第一列（输入）和第二列（输出）
data = []
with open(data_args.training_data_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        data.append({
            "vanilla adversarial": row["vanilla"] + row["adversarial"],
            "completion": row["completion"]
        })

random.shuffle(data)
# 若需要，可对数据进行截断，比如只使用前 1000 条
# data = data[:1000]

# 创建自定义数据集
custom_dataset = CustomDataset(data, tokenizer=tokenizer)

# 使用自定义 collator
my_collator = MyCollator(
    tokenizer=tokenizer,
    pad_token_id=tokenizer.pad_token_id,
    label_pad_token_id=-100
)

# 构造 Trainer 对象
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=custom_dataset,
    data_collator=my_collator,
)

# 开始训练并保存模型
trainer.train()
trainer.save_model(training_args.output_dir)
model.config.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
