# examples/train_example.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import logging
from pathlib import Path
import time
import json
from typing import Dict, List, Any, Optional
import numpy as np
from tqdm import tqdm

from custom_attention.modeling import CustomAttentionModel, AttentionConfig
from custom_attention.utils.helpers import setup_logger

class CustomDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, data_path: str, max_seq_length: int = 512):
        self.max_seq_length = max_seq_length
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载数据"""
        # 这里需要根据您的具体任务实现数据加载
        # 这里使用一个简单的示例
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # 这里需要根据您的具体任务实现数据预处理
        # 这里使用一个简单的示例
        input_ids = torch.tensor(item['input_ids'][:self.max_seq_length])
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor(item['labels'][:self.max_seq_length])
        
        # 添加padding
        if len(input_ids) < self.max_seq_length:
            padding = torch.zeros(self.max_seq_length - len(input_ids), dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
            attention_mask = torch.cat([attention_mask, torch.zeros_like(padding)])
            labels = torch.cat([labels, torch.zeros_like(padding)])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class Trainer:
    """训练器类"""
    def __init__(
        self,
        model: CustomAttentionModel,
        train_config: Dict[str, Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.config = train_config
        self.logger = setup_logger(__name__)
        
        # 设置优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
        
        # 设置学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_config['num_epochs']
        )
        
        # 设置损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 设置TensorBoard
        self.writer = SummaryWriter(log_dir=train_config['log_dir'])
        
        # 设置早停
        self.best_val_loss = float('inf')
        self.patience = train_config['patience']
        self.patience_counter = 0
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_steps = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # 将数据移到设备上
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            outputs = self.model(**batch)
            loss = self.criterion(outputs['last_hidden_state'], batch['labels'])
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['max_grad_norm']
            )
            
            self.optimizer.step()
            
            # 更新统计信息
            total_loss += loss.item()
            total_steps += 1
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
        
        return {
            'train_loss': total_loss / total_steps
        }
    
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = self.criterion(outputs['last_hidden_state'], batch['labels'])
                
                total_loss += loss.item()
                total_steps += 1
        
        return {
            'eval_loss': total_loss / total_steps
        }
    
    def train(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        save_dir: str
    ) -> Dict[str, List[float]]:
        """完整的训练流程"""
        history = {
            'train_loss': [],
            'eval_loss': []
        }
        
        for epoch in range(self.config['num_epochs']):
            self.logger.info(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['train_loss'])
            
            # 评估
            eval_metrics = self.evaluate(eval_loader)
            history['eval_loss'].append(eval_metrics['eval_loss'])
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录到TensorBoard
            self.writer.add_scalar('Loss/train', train_metrics['train_loss'], epoch)
            self.writer.add_scalar('Loss/eval', eval_metrics['eval_loss'], epoch)
            self.writer.add_scalar('LR', self.scheduler.get_last_lr()[0], epoch)
            
            # 保存最佳模型
            if eval_metrics['eval_loss'] < self.best_val_loss:
                self.best_val_loss = eval_metrics['eval_loss']
                self.patience_counter = 0
                self.save_checkpoint(save_dir, epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            # 定期保存检查点
            if (epoch + 1) % self.config['save_steps'] == 0:
                self.save_checkpoint(save_dir, epoch)
            
            # 早停
            if self.patience_counter >= self.patience:
                self.logger.info("Early stopping triggered")
                break
        
        self.writer.close()
        return history
    
    def save_checkpoint(self, save_dir: str, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        # 保存最新检查点
        checkpoint_path = Path(save_dir) / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            best_model_path = Path(save_dir) / 'best_model.pt'
            torch.save(checkpoint, best_model_path)
            self.logger.info(f"保存最佳模型到 {best_model_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练Attention模型")
    parser.add_argument('--config', type=str, required=True, help='训练配置文件路径')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 设置日志
    logger = setup_logger(__name__)
    
    try:
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化模型
        model_config = AttentionConfig(**config['model_config'])
        model = CustomAttentionModel(model_config)
        
        # 准备数据
        train_dataset = CustomDataset(
            Path(args.data_dir) / 'train.json',
            max_seq_length=config['max_seq_length']
        )
        eval_dataset = CustomDataset(
            Path(args.data_dir) / 'eval.json',
            max_seq_length=config['max_seq_length']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers']
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )
        
        # 初始化训练器
        trainer = Trainer(model, config['train_config'])
        
        # 开始训练
        history = trainer.train(train_loader, eval_loader, args.output_dir)
        
        # 保存训练历史
        with open(output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info("训练完成")
        
    except Exception as e:
        logger.error(f"训练过程出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()




# python train_example.py \
#     --config train_config.json \
#     --data_dir /path/to/data \
#     --output_dir /path/to/output