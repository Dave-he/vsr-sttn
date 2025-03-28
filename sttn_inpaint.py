import torch
import cv2
import numpy as np
import copy
from config import STTN_NEIGHBOR_STRIDE, STTN_REFERENCE_LENGTH, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT
from models import InpaintGenerator
from data_processing import _to_tensors


class STTNInpaint:
    def __init__(self):
        if torch.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        try:
            self.model = InpaintGenerator().to(self.device)
            self.model.load_state_dict(torch.load(
                'models/sttn_model.pth', map_location=self.device)['netG'])
            self.model.eval()
        except FileNotFoundError:
            print("Error: Model file 'models/sttn_model.pth' not found!")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        self.model_input_width = MODEL_INPUT_WIDTH
        self.model_input_height = MODEL_INPUT_HEIGHT
        self.neighbor_stride = STTN_NEIGHBOR_STRIDE
        self.ref_length = STTN_REFERENCE_LENGTH

    def __call__(self, input_frames: list[np.ndarray], input_mask: np.ndarray):
        _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
        mask = mask[:, :, None]
        H_ori, W_ori = mask.shape[:2]
        H_ori = int(H_ori + 0.5)
        W_ori = int(W_ori + 0.5)
        split_h = int(W_ori * 3 / 16)
        inpaint_area = self.get_inpaint_area_by_mask(H_ori, split_h, mask)
        frames_hr = copy.deepcopy(input_frames)
        frames_scaled = {}
        comps = {}
        inpainted_frames = []
        for k in range(len(inpaint_area)):
            frames_scaled[k] = []

        for j in range(len(frames_hr)):
            image = frames_hr[j]
            for k in range(len(inpaint_area)):
                image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                image_resize = cv2.resize(
                    image_crop, (self.model_input_width, self.model_input_height))
                frames_scaled[k].append(image_resize)

        for k in range(len(inpaint_area)):
            comps[k] = self.inpaint(frames_scaled[k])

        if inpaint_area:
            for j in range(len(frames_hr)):
                frame = frames_hr[j]
                for k in range(len(inpaint_area)):
                    comp = cv2.resize(
                        comps[k][j], (W_ori, split_h))
                    comp = cv2.cvtColor(np.array(comp).astype(
                        np.uint8), cv2.COLOR_BGR2RGB)
                    mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]
                    frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = mask_area * comp + \
                        (1 - mask_area) * \
                        frame[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                inpainted_frames.append(frame)
        return inpainted_frames

    @staticmethod
    def read_mask(path):
        img = cv2.imread(path, 0)
        ret, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        img = img[:, :, None]
        return img

    def get_ref_index(self, neighbor_ids, length):
        ref_index = []
        for i in range(0, length, self.ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
        return ref_index

    def inpaint(self, frames: list[np.ndarray]):
        frame_length = len(frames)
        feats = _to_tensors(frames).unsqueeze(0) * 2 - 1
        feats = feats.to(self.device)
        comp_frames = [None] * frame_length
        with torch.no_grad():
            feats = self.model.encoder(feats.view(
                frame_length, 3, self.model_input_height, self.model_input_width))
            _, c, feat_h, feat_w = feats.size()
            feats = feats.view(1, frame_length, c, feat_h, feat_w)
        for f in range(0, frame_length, self.neighbor_stride):
            neighbor_ids = [i for i in range(max(
                0, f - self.neighbor_stride), min(frame_length, f + self.neighbor_stride + 1))]
            ref_ids = self.get_ref_index(neighbor_ids, frame_length)
            with torch.no_grad():
                pred_feat = self.model.infer(
                    feats[0, neighbor_ids + ref_ids, :, :, :])
                pred_img = torch.tanh(self.model.decoder(
                    pred_feat[:len(neighbor_ids), :, :, :])).detach()
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8)
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(
                            np.float32) * 0.5 + img.astype(np.float32) * 0.5
        return comp_frames

    @staticmethod
    def get_inpaint_area_by_mask(H, h, mask):
        inpaint_area = []
        to_H = from_H = H
        while from_H != 0:
            if to_H - h < 0:
                from_H = 0
                to_H = h
            else:
                from_H = to_H - h
            if not np.all(mask[from_H:to_H, :] == 0) and np.sum(mask[from_H:to_H, :]) > 10:
                if to_H != H:
                    move = 0
                    while to_H + move < H and not np.all(mask[to_H + move, :] == 0):
                        move += 1
                    if to_H + move < H and move < h:
                        to_H += move
                        from_H += move
                if (from_H, to_H) not in inpaint_area:
                    inpaint_area.append((from_H, to_H))
                else:
                    break
            to_H -= h
        return inpaint_area

    @staticmethod
    def get_inpaint_area_by_selection(input_sub_area, mask):
        print('use selection area for inpainting')
        height, width = mask.shape[:2]
        ymin, ymax, _, _ = input_sub_area
        interval_size = 135
        inpaint_area = []
        for i in range(ymin, ymax, interval_size):
            inpaint_area.append((i, i + interval_size))
        if inpaint_area[-1][1] != ymax:
            if inpaint_area[-1][1] + interval_size <= height:
                inpaint_area.append(
                    (inpaint_area[-1][1], inpaint_area[-1][1] + interval_size))
        return inpaint_area