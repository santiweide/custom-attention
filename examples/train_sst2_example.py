# examples/train_sst2_example.py

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from typing import Dict, List, Any

class SST2Dataset(Dataset):
    """SST-2数据集类"""
    def __init__(self, dataset, tokenizer, max_length: int = 128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        
        # 对文本进行编码
        encoding = self.tokenizer(
            item['sentence'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 将编码转换为张量
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # 获取标签
        label = torch.tensor(item['label'], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }

def prepare_sst2_data():
    """准备SST-2数据集"""
    # 加载数据集
    dataset = load_dataset("glue", "sst2")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 创建数据集
    train_dataset = SST2Dataset(dataset['train'], tokenizer)
    eval_dataset = SST2Dataset(dataset['validation'], tokenizer)
    
    return train_dataset, eval_dataset, tokenizer

def main():
    # 准备数据
    train_dataset, eval_dataset, tokenizer = prepare_sst2_data()
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # 配置模型
    config = {
        "model_config": {
            "hidden_size": 768,
            "num_attention_heads": 12,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 128,  # SST-2句子较短
            "vocab_size": tokenizer.vocab_size
        },
        "train_config": {
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "num_epochs": 3,
            "batch_size": 32,
            "max_grad_norm": 1.0,
            "patience": 3,
            "save_steps": 1000,
            "log_dir": "logs"
        }
    }
    
    # 初始化模型
    model = CustomAttentionModel(AttentionConfig(**config["model_config"]))
    
    # 初始化训练器
    trainer = Trainer(model, config["train_config"])
    
    # 开始训练
    history = trainer.train(train_loader, eval_loader, "output/sst2")
    
    print("训练完成！")

if __name__ == "__main__":
    main()