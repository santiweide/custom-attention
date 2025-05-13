# create_env.sh

#!/bin/bash

# 设置环境名称和Python版本
ENV_NAME="custom-attention"
PYTHON_VERSION="3.9"

# 创建conda环境
echo "创建conda环境: ${ENV_NAME}..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# 激活环境
echo "激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# 安装PyTorch (根据您的CUDA版本选择)
echo "安装PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装其他依赖
echo "安装其他依赖..."
pip install -r requirements.txt

# 验证安装
echo "验证安装..."
python -c "import torch; print('PyTorch版本:', torch.__version__)"
python -c "import torch; print('CUDA是否可用:', torch.cuda.is_available())"

echo "环境创建完成！"
echo "使用以下命令激活环境："
echo "conda activate ${ENV_NAME}"
