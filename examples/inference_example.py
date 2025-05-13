# examples/inference_example.py

import torch
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import time
import json

from custom_attention.modeling import CustomAttentionModel, AttentionConfig
from custom_attention.utils.helpers import setup_logger

class InferencePipeline:
    """推理流水线类"""
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 1,
        max_seq_length: int = 512
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.logger = setup_logger(__name__)
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()  # 设置为评估模式
        
    def _load_model(self, model_path: str) -> CustomAttentionModel:
        """加载模型"""
        try:
            self.logger.info(f"正在从 {model_path} 加载模型...")
            model = CustomAttentionModel.from_pretrained(model_path)
            model = model.to(self.device)
            self.logger.info("模型加载成功")
            return model
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise

    def preprocess(self, input_text: str) -> Dict[str, torch.Tensor]:
        """预处理输入文本"""
        # 这里需要根据您的具体任务实现tokenization
        # 这里使用一个简单的示例
        tokens = input_text.split()[:self.max_seq_length]
        input_ids = torch.tensor([hash(token) % 1000 for token in tokens])
        
        # 添加padding
        if len(input_ids) < self.max_seq_length:
            padding = torch.zeros(self.max_seq_length - len(input_ids), dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
        
        # 创建attention mask
        attention_mask = torch.ones_like(input_ids)
        attention_mask[len(tokens):] = 0
        
        return {
            "input_ids": input_ids.unsqueeze(0).to(self.device),  # 添加batch维度
            "attention_mask": attention_mask.unsqueeze(0).to(self.device)
        }

    @torch.no_grad()  # 禁用梯度计算
    def inference(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """执行推理"""
        try:
            start_time = time.time()
            
            # 执行模型推理
            outputs = self.model(**inputs)
            
            # 计算推理时间
            inference_time = time.time() - start_time
            
            return {
                "outputs": outputs,
                "inference_time": inference_time
            }
        except Exception as e:
            self.logger.error(f"推理过程出错: {str(e)}")
            raise

    def postprocess(self, model_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """后处理模型输出"""
        # 这里需要根据您的具体任务实现后处理逻辑
        # 这里使用一个简单的示例
        last_hidden_state = model_outputs["outputs"]["last_hidden_state"]
        
        # 将输出转换为CPU并转为numpy
        output_numpy = last_hidden_state.cpu().numpy()
        
        return {
            "processed_output": output_numpy.tolist(),
            "inference_time": model_outputs["inference_time"]
        }

    def run(self, input_text: str) -> Dict[str, Any]:
        """运行完整的推理流程"""
        try:
            # 预处理
            inputs = self.preprocess(input_text)
            
            # 推理
            model_outputs = self.inference(inputs)
            
            # 后处理
            results = self.postprocess(model_outputs)
            
            return results
        except Exception as e:
            self.logger.error(f"推理流程出错: {str(e)}")
            raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行Attention模型推理")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="预训练模型的路径"
    )
    parser.add_argument(
        "--input_text",
        type=str,
        required=True,
        help="输入文本"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="inference_results.json",
        help="输出结果的文件路径"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批处理大小"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="最大序列长度"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger(__name__)
    
    try:
        # 初始化推理流水线
        pipeline = InferencePipeline(
            model_path=args.model_path,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length
        )
        
        # 运行推理
        results = pipeline.run(args.input_text)
        
        # 保存结果
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"推理完成，结果已保存到 {args.output_file}")
        
    except Exception as e:
        logger.error(f"推理过程出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()