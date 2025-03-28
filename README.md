# VSR-STTN 视频去字幕工具

基于STTN模型的视频去字幕工具，能够有效去除视频中的字幕并保持画面连贯性。

### 工程特点
1. 模块化设计：核心功能分离到独立模块
2. 类型提示：使用Python类型提示提高代码可读性
3. 错误处理：完善的异常处理机制
4. 性能优化：支持批量处理和GPU加速
5. 用户友好：提供详细的文档和命令行参数

### 依赖管理
- 使用PyTorch作为深度学习框架
- 使用OpenCV进行视频处理
- 使用tqdm显示进度条
- 支持CUDA和MPS加速

## 工程目录
```
vsr-sttn/
├── config.py
├── data_processing.py
├── models.py
├── sttn_inpaint.py
├── sttn_video_inpaint.py
├── subtitle_remover.py
└── main.py
```

# 前置依赖 

1. 安装 ffmpeg
2. 安装 python 环境
3. 安装 pytorch 环境
4. 安装 opencv 环境

## 安装指南
1. 克隆仓库：`git clone https://github.com/Dave-he/vsr-sttn`
2. 安装依赖：`pip install -r requirements.txt`

# 预训练模型
- 模型文件： models/sttn_model.pth

# 安装依赖
  ```shell
conda create -n vsr python=3.12.7
conda activate vsr
pip install torch torchvision numpy
#pip install --upgrade opencv-contrib-python #安装扩展包，不然DualTVL1OpticalFlow_create找不到
  ```

## 使用方法
```bash
python main.py
```