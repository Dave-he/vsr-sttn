import multiprocessing
import time
from tqdm import tqdm
import cv2
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from typing import List
import math
import copy


SUBTITLE_AREA_DEVIATION_PIXEL = 10

class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        for i in range(len(img_group)):
            if img_group[i].ndim == 3:
                img_array = cv2.cvtColor(img_group[i], cv2.COLOR_BGR2RGB)
                img_group[i] = Image.fromarray(img_array)
            elif img_group[i].ndim == 2:
                img_group[i] = Image.fromarray(img_group[i])

        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        p_attn = F.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.attention = Attention()

    def forward(self, x, b, c):
        bt, _, h, w = x.size()
        t = bt // b
        d_k = c // len(self.patchsize)
        output = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)
        for (width, height), query, key, value in zip(self.patchsize,
                                                      torch.chunk(_query, len(self.patchsize), dim=1), torch.chunk(
                                                          _key, len(self.patchsize), dim=1),
                                                      torch.chunk(_value, len(self.patchsize), dim=1)):
            out_w, out_h = w // width, h // height

            # 1) embedding and reshape
            query = query.view(b, t, d_k, out_h, height, out_w, width)
            query = query.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            key = key.view(b, t, d_k, out_h, height, out_w, width)
            key = key.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            value = value.view(b, t, d_k, out_h, height, out_w, width)
            value = value.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            '''
            # 2) Apply attention on all the projected vectors in batch.
            tmp1 = []
            for q,k,v in zip(torch.chunk(query, b, dim=0), torch.chunk(key, b, dim=0), torch.chunk(value, b, dim=0)):
                y, _ = self.attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
                tmp1.append(y)
            y = torch.cat(tmp1,1)
            '''
            y, _ = self.attention(query, key, value)
            # 3) "Concat" using a view and apply a final linear.
            y = y.view(b, t, out_h, out_w, d_k, height, width)
            y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)
            output.append(y)
        output = torch.cat(output, 1)
        x = self.output_linear(output)
        return x


# Standard 2 layerd FFN of transformer
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, patchsize, hidden=128):
        super().__init__()
        self.attention = MultiHeadedAttention(patchsize, d_model=hidden)
        self.feed_forward = FeedForward(hidden)

    def forward(self, x):
        x, b, c = x['x'], x['b'], x['c']
        x = x + self.attention(x, b, c)
        x = x + self.feed_forward(x)
        return {'x': x, 'b': b, 'c': c}


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class InpaintGenerator(BaseNetwork):
    def __init__(self, init_weights=True):
        super(InpaintGenerator, self).__init__()
        channel = 256
        stack_num = 8
        patchsize = [(80, 15), (32, 6), (10, 5), (5, 3)]
        blocks = []
        for _ in range(stack_num):
            blocks.append(TransformerBlock(patchsize, hidden=channel))
        self.transformer = nn.Sequential(*blocks)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # decoder: decode frames from features
        self.decoder = nn.Sequential(
            Deconv(channel, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            Deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

        if init_weights:
            self.init_weights()

    def forward(self, masked_frames):
        # extracting features
        b, t, c, h, w = masked_frames.size()
        enc_feat = self.encoder(masked_frames.view(b*t, c, h, w))
        _, c, h, w = enc_feat.size()
        enc_feat = self.transformer(
            {'x': enc_feat, 'b': b, 'c': c})['x']
        output = self.decoder(enc_feat)
        output = torch.tanh(output)
        return output

    def infer(self, feat):
        t, c, _, _ = feat.size()
        enc_feat = self.transformer(
            {'x': feat, 'b': 1, 'c': c})['x']
        return enc_feat


# 定义图像预处理方式
_to_tensors = transforms.Compose([
    Stack(),  # 将图像堆叠为序列
    ToTorchFormatTensor()  # 将堆叠的图像转化为PyTorch张量
])

# 参考帧步长
STTN_NEIGHBOR_STRIDE = 5
# 参考帧长度（数量）
STTN_REFERENCE_LENGTH = 10


class STTNInpaint:
    def __init__(self):
        # 检查是否支持MPS
        if torch.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # 1. 创建InpaintGenerator模型实例并装载到选择的设备上
        self.model = InpaintGenerator().to(self.device)
        # 2. 载入预训练模型的权重，转载模型的状态字典
        self.model.load_state_dict(torch.load(
            'models/sttn_model.pth', map_location=self.device)['netG'])

        # 3. # 将模型设置为评估模式
        self.model.eval()
        # 模型输入用的宽和高
        self.model_input_width, self.model_input_height = 640, 120
        # 2. 设置相连帧数
        self.neighbor_stride = STTN_NEIGHBOR_STRIDE
        self.ref_length = STTN_REFERENCE_LENGTH

    def __call__(self, input_frames: List[np.ndarray], input_mask: np.ndarray):
        """
        :param input_frames: 原视频帧
        :param mask: 字幕区域mask
        """
        _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
        mask = mask[:, :, None]
        H_ori, W_ori = mask.shape[:2]
        H_ori = int(H_ori + 0.5)
        W_ori = int(W_ori + 0.5)
        # 确定去字幕的垂直高度部分
        split_h = int(W_ori * 3 / 16)
        inpaint_area = self.get_inpaint_area_by_mask(H_ori, split_h, mask)
        # 初始化帧存储变量
        # 高分辨率帧存储列表
        frames_hr = copy.deepcopy(input_frames)
        frames_scaled = {}  # 存放缩放后帧的字典
        comps = {}  # 存放补全后帧的字典
        # 存储最终的视频帧
        inpainted_frames = []
        for k in range(len(inpaint_area)):
            frames_scaled[k] = []  # 为每个去除部分初始化一个列表

        # 读取并缩放帧
        for j in range(len(frames_hr)):
            image = frames_hr[j]
            # 对每个去除部分进行切割和缩放
            for k in range(len(inpaint_area)):
                image_crop = image[inpaint_area[k][0]
                    :inpaint_area[k][1], :, :]  # 切割
                image_resize = cv2.resize(
                    # 缩放
                    image_crop, (self.model_input_width, self.model_input_height))
                frames_scaled[k].append(image_resize)  # 将缩放后的帧添加到对应列表

        # 处理每一个去除部分
        for k in range(len(inpaint_area)):
            # 调用inpaint函数进行处理
            comps[k] = self.inpaint(frames_scaled[k])

        # 如果存在去除部分
        if inpaint_area:
            for j in range(len(frames_hr)):
                frame = frames_hr[j]  # 取出原始帧
                # 对于模式中的每一个段落
                for k in range(len(inpaint_area)):
                    comp = cv2.resize(
                        comps[k][j], (W_ori, split_h))  # 将补全帧缩放回原大小
                    comp = cv2.cvtColor(np.array(comp).astype(
                        np.uint8), cv2.COLOR_BGR2RGB)  # 转换颜色空间
                    # 获取遮罩区域并进行图像合成
                    mask_area = mask[inpaint_area[k][0]
                        :inpaint_area[k][1], :]  # 取出遮罩区域
                    # 实现遮罩区域内的图像融合
                    frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = mask_area * comp + \
                        (1 - mask_area) * \
                        frame[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                # 将最终帧添加到列表
                inpainted_frames.append(frame)
                print(f'processing frame, {len(frames_hr) - j} left')
        return inpainted_frames

    @staticmethod
    def read_mask(path):
        img = cv2.imread(path, 0)
        # 转为binary mask
        ret, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        img = img[:, :, None]
        return img

    def get_ref_index(self, neighbor_ids, length):
        """
        采样整个视频的参考帧
        """
        # 初始化参考帧的索引列表
        ref_index = []
        # 在视频长度范围内根据ref_length逐步迭代
        for i in range(0, length, self.ref_length):
            # 如果当前帧不在近邻帧中
            if i not in neighbor_ids:
                # 将它添加到参考帧列表
                ref_index.append(i)
        # 返回参考帧索引列表
        return ref_index

    def inpaint(self, frames: List[np.ndarray]):
        """
        使用STTN完成空洞填充（空洞即被遮罩的区域）
        """
        frame_length = len(frames)
        # 对帧进行预处理转换为张量，并进行归一化
        feats = _to_tensors(frames).unsqueeze(0) * 2 - 1
        # 把特征张量转移到指定的设备（CPU或GPU）
        feats = feats.to(self.device)
        # 初始化一个与视频长度相同的列表，用于存储处理完成的帧
        comp_frames = [None] * frame_length
        # 关闭梯度计算，用于推理阶段节省内存并加速
        with torch.no_grad():
            # 将处理好的帧通过编码器，产生特征表示
            feats = self.model.encoder(feats.view(
                frame_length, 3, self.model_input_height, self.model_input_width))
            # 获取特征维度信息
            _, c, feat_h, feat_w = feats.size()
            # 调整特征形状以匹配模型的期望输入
            feats = feats.view(1, frame_length, c, feat_h, feat_w)
        # 获取重绘区域
        # 在设定的邻居帧步幅内循环处理视频
        for f in range(0, frame_length, self.neighbor_stride):
            # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "inpaint comps:", f + 1, '/', frame_length)
            # 计算邻近帧的ID
            neighbor_ids = [i for i in range(max(
                0, f - self.neighbor_stride), min(frame_length, f + self.neighbor_stride + 1))]
            # 获取参考帧的索引
            ref_ids = self.get_ref_index(neighbor_ids, frame_length)
            # 同样关闭梯度计算
            with torch.no_grad():
                # 通过模型推断特征并传递给解码器以生成完成的帧
                pred_feat = self.model.infer(
                    feats[0, neighbor_ids + ref_ids, :, :, :])
                # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "inpaint infer:", f + 1, '/', frame_length)
                # 将预测的特征通过解码器生成图片，并应用激活函数tanh，然后分离出张量
                pred_img = torch.tanh(self.model.decoder(
                    pred_feat[:len(neighbor_ids), :, :, :])).detach()
                # 将结果张量重新缩放到0到255的范围内（图像像素值）
                pred_img = (pred_img + 1) / 2
                # 将张量移动回CPU并转为NumPy数组
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "inpaint tanh:", f + 1, '/', frame_length)
                # 遍历邻近帧
                for i in range(len(neighbor_ids)):
                    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "inpainted frame:", f + i + 1, '/', frame_length)
                    idx = neighbor_ids[i]
                    # 将预测的图片转换为无符号8位整数格式
                    img = np.array(pred_img[i]).astype(np.uint8)
                    if comp_frames[idx] is None:
                        # 如果该位置为空，则赋值为新计算出的图片
                        comp_frames[idx] = img
                    else:
                        # 如果此位置之前已有图片，则将新旧图片混合以提高质量
                        comp_frames[idx] = comp_frames[idx].astype(
                            np.float32) * 0.5 + img.astype(np.float32) * 0.5
        # 返回处理完成的帧序列
        return comp_frames

    @staticmethod
    def get_inpaint_area_by_mask(H, h, mask):
        """
        获取字幕去除区域，根据mask来确定需要填补的区域和高度
        """
        # 存储绘画区域的列表
        inpaint_area = []
        # 从视频底部的字幕位置开始，假设字幕通常位于底部
        to_H = from_H = H
        # 从底部向上遍历遮罩
        while from_H != 0:
            if to_H - h < 0:
                # 如果下一段会超出顶端，则从顶端开始
                from_H = 0
                to_H = h
            else:
                # 确定段的上边界
                from_H = to_H - h
            # 检查当前段落是否包含遮罩像素
            if not np.all(mask[from_H:to_H, :] == 0) and np.sum(mask[from_H:to_H, :]) > 10:
                # 如果不是第一个段落，向下移动以确保没遗漏遮罩区域
                if to_H != H:
                    move = 0
                    while to_H + move < H and not np.all(mask[to_H + move, :] == 0):
                        move += 1
                    # 确保没有越过底部
                    if to_H + move < H and move < h:
                        to_H += move
                        from_H += move
                # 将该段落添加到列表中
                if (from_H, to_H) not in inpaint_area:
                    inpaint_area.append((from_H, to_H))
                else:
                    break
            # 移动到下一个段落
            to_H -= h
        return inpaint_area  # 返回绘画区域列表

    @staticmethod
    def get_inpaint_area_by_selection(input_sub_area, mask):
        print('use selection area for inpainting')
        height, width = mask.shape[:2]
        ymin, ymax, _, _ = input_sub_area
        interval_size = 135
        # 存储结果的列表
        inpaint_area = []
        # 计算并存储标准区间
        for i in range(ymin, ymax, interval_size):
            inpaint_area.append((i, i + interval_size))
        # 检查最后一个区间是否达到了最大值
        if inpaint_area[-1][1] != ymax:
            # 如果没有，则创建一个新的区间，开始于最后一个区间的结束，结束于扩大后的值
            if inpaint_area[-1][1] + interval_size <= height:
                inpaint_area.append(
                    (inpaint_area[-1][1], inpaint_area[-1][1] + interval_size))
        return inpaint_area  # 返回绘画区域列表


class STTNVideoInpaint:

    def read_frame_info_from_video(self):
        # 使用opencv读取视频
        reader = cv2.VideoCapture(self.video_path)
        # 获取视频的宽度, 高度, 帧率和帧数信息并存储在frame_info字典中
        frame_info = {
            # 视频的原始宽度
            'W_ori': int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),
            # 视频的原始高度
            'H_ori': int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5),
            'fps': reader.get(cv2.CAP_PROP_FPS),  # 视频的帧率
            'len': int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)  # 视频的总帧数
        }
        # 返回视频读取对象、帧信息和视频写入对象
        return reader, frame_info

    def __init__(self, video_path, mask_path=None):
        # STTNInpaint视频修复实例初始化
        self.sttn_inpaint = STTNInpaint()
        # 可在一次处理中加载的最大帧数
        self.clip_gap = 50
        # 视频和掩码路径
        self.video_path = video_path
        self.mask_path = mask_path
        # 设置输出视频文件的路径
        self.video_out_path = os.path.join(
            os.path.dirname(os.path.abspath(self.video_path)),
            f"{os.path.basename(self.video_path).rsplit('.', 1)[0]}_no_sub.mp4"
        )

    def __call__(self, input_mask=None, input_sub_remover=None, tbar=None):
        # 读取视频帧信息
        reader, frame_info = self.read_frame_info_from_video()
        if input_sub_remover is not None:
            writer = input_sub_remover.video_writer
        else:
            # 创建视频写入对象，用于输出修复后的视频
            writer = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(
                *"mp4v"), frame_info['fps'], (frame_info['W_ori'], frame_info['H_ori']))
        # 计算需要迭代修复视频的次数
        rec_time = frame_info['len'] // self.clip_gap if frame_info['len'] % self.clip_gap == 0 else frame_info['len'] // self.clip_gap + 1
        # 计算分割高度，用于确定修复区域的大小
        split_h = int(frame_info['W_ori'] * 3 / 16)
        if input_mask is None:
            # 读取掩码
            mask = self.sttn_inpaint.read_mask(self.mask_path)
        else:
            _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
            mask = mask[:, :, None]
        # 得到修复区域位置
        inpaint_area = self.sttn_inpaint.get_inpaint_area_by_mask(
            frame_info['H_ori'], split_h, mask)
        # 遍历每一次的迭代次数
        for i in range(rec_time):
            start_f = i * self.clip_gap  # 起始帧位置
            end_f = min((i + 1) * self.clip_gap, frame_info['len'])  # 结束帧位置
            print(time.time(), 'Processing:', start_f + 1,
                  '-', end_f, ' / Total:', frame_info['len'])
            frames_hr = []  # 高分辨率帧列表
            frames = {}  # 帧字典，用于存储裁剪后的图像
            comps = {}  # 组合字典，用于存储修复后的图像
            # 初始化帧字典
            for k in range(len(inpaint_area)):
                frames[k] = []
            # 读取和修复高分辨率帧
            for j in range(start_f, end_f):
                success, image = reader.read()
                frames_hr.append(image)
                for k in range(len(inpaint_area)):
                    # 裁剪、缩放并添加到帧字典
                    image_crop = image[inpaint_area[k]
                                       [0]:inpaint_area[k][1], :, :]
                    image_resize = cv2.resize(
                        image_crop, (self.sttn_inpaint.model_input_width, self.sttn_inpaint.model_input_height))
                    frames[k].append(image_resize)
            # 对每个修复区域运行修复
            for k in range(len(inpaint_area)):
                # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "inpaint comps:", i + 1, '/', rec_time, k)
                comps[k] = self.sttn_inpaint.inpaint(frames[k])
            # 如果有要修复的区域
            if inpaint_area is not []:
                for j in range(end_f - start_f):
                    if input_sub_remover is not None and input_sub_remover.gui_mode:
                        original_frame = copy.deepcopy(frames_hr[j])
                    else:
                        original_frame = None
                    frame = frames_hr[j]
                    for k in range(len(inpaint_area)):
                        # 将修复的图像重新扩展到原始分辨率，并融合到原始帧
                        comp = cv2.resize(
                            comps[k][j], (frame_info['W_ori'], split_h))
                        comp = cv2.cvtColor(np.array(comp).astype(
                            np.uint8), cv2.COLOR_BGR2RGB)
                        mask_area = mask[inpaint_area[k]
                                         [0]:inpaint_area[k][1], :]
                        frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = mask_area * comp + \
                            (1 - mask_area) * \
                            frame[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                    writer.write(frame)
                    if input_sub_remover is not None:
                        if tbar is not None:
                            input_sub_remover.update_progress(
                                tbar, increment=1)
                        if original_frame is not None and input_sub_remover.gui_mode:
                            input_sub_remover.preview_frame = cv2.hconcat(
                                [original_frame, frame])
        # 释放视频写入对象
        writer.release()


def create_mask(size, coords_list):
    mask = np.zeros(size, dtype="uint8")
    if coords_list:
        for coords in coords_list:
            xmin, xmax, ymin, ymax = coords
            # 为了避免框过小，放大10个像素
            x1 = xmin - SUBTITLE_AREA_DEVIATION_PIXEL
            if x1 < 0:
                x1 = 0
            y1 = ymin - SUBTITLE_AREA_DEVIATION_PIXEL
            if y1 < 0:
                y1 = 0
            x2 = xmax + SUBTITLE_AREA_DEVIATION_PIXEL
            y2 = ymax + SUBTITLE_AREA_DEVIATION_PIXEL
            cv2.rectangle(mask, (x1, y1),
                          (x2, y2), (255, 255, 255), thickness=-1)
    return mask


class SubtitleRemover:
    def __init__(self, vd_path, sub_area=None, gui_mode=False):
        # 线程锁
        self.lock = threading.RLock()
        # 用户指定的字幕区域位置
        self.sub_area = sub_area
        # 是否为gui运行，gui运行需要显示预览
        self.gui_mode = gui_mode
        # 判断是否为图片
        self.is_picture = False

        # 视频路径
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(vd_path)
        # 通过视频路径获取视频名称
        self.vd_name = Path(self.video_path).stem
        # 视频帧总数
        self.frame_count = int(self.video_cap.get(
            cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        # 视频帧率
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # 视频尺寸
        self.size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.mask_size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 创建字幕检测对象
        # self.sub_detector = SubtitleDetect(self.video_path, self.sub_area)
        # 创建视频临时对象，windows下delete=True会有permission denied的报错
        self.video_temp_file = tempfile.NamedTemporaryFile(
            suffix='.mp4', delete=False)
        # 创建视频写对象
        self.video_writer = cv2.VideoWriter(
            self.video_temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
        self.video_out_name = os.path.join(os.path.dirname(
            self.video_path), f'{self.vd_name}_no_sub.mp4')
        self.video_inpaint = None
        self.lama_inpaint = None
        self.ext = os.path.splitext(vd_path)[-1]

        if torch.mps.is_available():
            print('use MPS for acceleration')
        elif torch.distributed.is_available():
            print('use DDP for acceleration')
        elif torch.cuda.is_available():
            print('use GPU for acceleration')
        else:
            print('use CPU for acceleration')

        # 总处理进度
        self.progress_total = 0
        self.progress_remover = 0
        self.isFinished = False
        # 预览帧
        self.preview_frame = None
        # 是否将原音频嵌入到去除字幕后的视频
        self.is_successful_merged = False

    @staticmethod
    def get_coordinates(dt_box):
        """
        从返回的检测框中获取坐标
        :param dt_box 检测框返回结果
        :return list 坐标点列表
        """
        coordinate_list = list()
        if isinstance(dt_box, list):
            for i in dt_box:
                i = list(i)
                (x1, y1) = int(i[0][0]), int(i[0][1])
                (x2, y2) = int(i[1][0]), int(i[1][1])
                (x3, y3) = int(i[2][0]), int(i[2][1])
                (x4, y4) = int(i[3][0]), int(i[3][1])
                xmin = max(x1, x4)
                xmax = min(x2, x3)
                ymin = max(y1, y2)
                ymax = min(y3, y4)
                coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list

    @staticmethod
    def is_current_frame_no_start(frame_no, continuous_frame_no_list):
        """
        判断给定的帧号是否为开头，是的话返回结束帧号，不是的话返回-1
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no == frame_no:
                return True
        return False

    @staticmethod
    def find_frame_no_end(frame_no, continuous_frame_no_list):
        """
        判断给定的帧号是否为开头，是的话返回结束帧号，不是的话返回-1
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no <= frame_no <= end_no:
                return end_no
        return -1

    def update_progress(self, tbar, increment):
        tbar.update(increment)
        current_percentage = (tbar.n / tbar.total) * 100
        self.progress_remover = int(current_percentage) // 2
        self.progress_total = 50 + self.progress_remover

    def sttn_mode_with_no_detection(self, tbar):
        """
        使用sttn对选中区域进行重绘，不进行字幕检测
        """
        print('use sttn mode with no detection')
        print('[Processing] start removing subtitles...')
        if self.sub_area is not None:
            ymin, ymax, xmin, xmax = self.sub_area
        else:
            print('[Info] No subtitle area has been set. Video will be processed in full screen. As a result, the final outcome might be suboptimal.')
            ymin, ymax, xmin, xmax = 0, self.frame_height, 0, self.frame_width

        mask_area_coordinates = [(xmin, xmax, ymin, ymax)]
        mask = create_mask(self.mask_size, mask_area_coordinates)
        sttn_video_inpaint = STTNVideoInpaint(self.video_path)
        sttn_video_inpaint(input_mask=mask, input_sub_remover=self, tbar=tbar)

    def run(self):
        # 记录开始时间
        start_time = time.time()
        # 重置进度条
        self.progress_total = 0
        tbar = tqdm(total=int(self.frame_count), unit='frame', position=0, file=sys.__stdout__,
                    desc='Subtitle Removing')

        self.sttn_mode_with_no_detection(tbar)
        self.video_cap.release()
        self.video_writer.release()
        if not self.is_picture:
            # 将原音频合并到新生成的视频文件中
            self.merge_audio_to_video()
            print(
                f"[Finished]Subtitle successfully removed, video generated at：{self.video_out_name}")
        else:
            print(
                f"[Finished]Subtitle successfully removed, picture generated at：{self.video_out_name}")
        print(f'time cost: {round(time.time() - start_time, 2)}s')
        self.isFinished = True
        self.progress_total = 100
        if os.path.exists(self.video_temp_file.name):
            try:
                os.remove(self.video_temp_file.name)
            except Exception:
                if platform.system() in ['Windows']:
                    pass
                else:
                    print(
                        f'failed to delete temp file {self.video_temp_file.name}')

    def merge_audio_to_video(self):
        # 创建音频临时对象，windows下delete=True会有permission denied的报错
        temp = tempfile.NamedTemporaryFile(suffix='.aac', delete=False)
        audio_extract_command = ["ffmpeg",
                                 "-y", "-i", self.video_path,
                                 "-acodec", "copy",
                                 "-vn", "-loglevel", "error", temp.name]
        use_shell = True if os.name == "nt" else False
        try:
            subprocess.check_output(
                audio_extract_command, stdin=open(os.devnull), shell=use_shell)
        except Exception:
            print('fail to extract audio')
            return
        else:
            if os.path.exists(self.video_temp_file.name):
                audio_merge_command = ["ffmpeg",
                                       "-y", "-i", self.video_temp_file.name,
                                       "-i", temp.name,
                                       "-vcodec", "libx264",  # 使用h264编码
                                       "-acodec", "copy",
                                       "-loglevel", "error", self.video_out_name]
                try:
                    subprocess.check_output(
                        audio_merge_command, stdin=open(os.devnull), shell=use_shell)
                except Exception:
                    print('fail to merge audio')
                    return
            if os.path.exists(temp.name):
                try:
                    os.remove(temp.name)
                except Exception:
                    if platform.system() in ['Windows']:
                        pass
                    else:
                        print(f'failed to delete temp file {temp.name}')
            self.is_successful_merged = True
        finally:
            temp.close()
            if not self.is_successful_merged:
                try:
                    shutil.copy2(self.video_temp_file.name,
                                 self.video_out_name)
                except IOError as e:
                    print("Unable to copy file. %s" % e)
            self.video_temp_file.close()


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    # 1. 提示用户输入视频路径
    video_path = input(f"Please input video or image file path: ").strip()
    sub_area = (700, 1200, 20, 700)
    SubtitleRemover(video_path, sub_area=sub_area).run()
