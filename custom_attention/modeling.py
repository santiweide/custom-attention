# custom_attention/modeling.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

class AttentionConfig:
    """Attention模型的配置类"""
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        layer_norm_eps: float = 1e-12,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        for key, value in kwargs.items():
            setattr(self, key, value)

class AttentionOutput(nn.Module):
    """Attention输出层"""
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states

class CustomAttention(nn.Module):
    """自定义Attention实现"""
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义Q、K、V的线性变换
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output = AttentionOutput(config)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """重塑张量以便进行多头注意力计算"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 计算Q、K、V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32))

        # 应用attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 应用softmax获取注意力权重
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # 计算输出
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # 重塑输出张量
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 通过输出层
        output = self.output(context_layer, hidden_states)

        if output_attentions:
            return output, attention_probs
        return output, None

class CustomAttentionModel(nn.Module):
    """完整的Attention模型"""
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        
        # 位置编码
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, 
            config.hidden_size
        )
        
        # 多头注意力层
        self.attention = CustomAttention(config)
        
        # 前馈网络
        self.intermediate = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        
        # 输出层
        self.output = AttentionOutput(config)
        
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_extended_attention_mask(
        self, 
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, ...],
        device: torch.device
    ) -> torch.Tensor:
        """扩展attention mask"""
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Attention mask should be 2D or 3D, got {attention_mask.dim()}D")

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask.to(device)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            input_ids: 输入序列的token ids
            attention_mask: 注意力掩码
            position_ids: 位置编码的ids
            output_attentions: 是否输出注意力权重
            return_dict: 是否返回字典格式的输出
            
        Returns:
            模型输出，包含隐藏状态和可选的注意力权重
        """
        batch_size, seq_length = input_ids.shape
        
        # 生成位置编码
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 获取位置嵌入
        position_embeddings = self.position_embeddings(position_ids)
        
        # 准备attention mask
        if attention_mask is not None:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, 
                (batch_size, seq_length),
                input_ids.device
            )
        else:
            extended_attention_mask = None

        # 注意力层
        attention_output, attention_probs = self.attention(
            input_ids + position_embeddings,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
        )

        # 前馈网络
        intermediate_output = self.intermediate(attention_output)
        
        # 输出层
        output = self.output(intermediate_output, attention_output)

        if not return_dict:
            return (output, attention_probs) if output_attentions else (output,)

        return {
            "last_hidden_state": output,
            "attention_probs": attention_probs if output_attentions else None,
        }

    def save_pretrained(self, save_directory: str):
        """保存模型"""
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")
        self.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        """加载预训练模型"""
        config = AttentionConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        model.load_state_dict(torch.load(f"{pretrained_model_name_or_path}/pytorch_model.bin"))
        return model