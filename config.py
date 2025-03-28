# config.py
import torch
from torchvision import transforms

SUBTITLE_AREA_DEVIATION_PIXEL = 10

# ------------
# 模型参数
# ------------
MODEL_PATH = 'models/sttn_model.pth'
MODEL_INPUT_WIDTH = 640
MODEL_INPUT_HEIGHT = 120
INIT_TYPE = 'normal'
INIT_GAIN = 0.02

# ------------
# 视频处理参数
# ------------
STTN_NEIGHBOR_STRIDE = 5
STTN_REFERENCE_LENGTH = 10
CLIP_GAP = 50  # 视频分块处理的间隔帧数
MASK_SPLIT_RATIO = 3/16  # 分割高度占宽度的比例
SKIP_FRAME_DIFF= 20  # 片段前后增加20帧

# ------------
# 路径配置
# ------------
DEFAULT_OUTPUT_SUFFIX = '_no_sub.mp4'
TEMP_FILE_PREFIX = 'vsr_sttn_temp_'


# ------------
# 预处理配置
# ------------
IMAGE_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH)),
    transforms.ToTensor(),
])

# ------------
# 硬件配置
# ------------
DEVICE = 'cuda' if torch.cuda.is_available() \
    else 'mps' if torch.mps.is_available() \
    else 'cpu'

# ------------
# 性能优化参数
# ------------
MAX_FRAMES_IN_MEMORY = 100  # 内存中同时处理的最大帧数
BATCH_SIZE = 8