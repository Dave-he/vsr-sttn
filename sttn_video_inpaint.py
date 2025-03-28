import cv2
import os
import time
import numpy as np
from tqdm import tqdm
from sttn_inpaint import STTNInpaint


class STTNVideoInpaint:
    def __init__(self, video_path, mask_path=None):
        self.sttn_inpaint = STTNInpaint()
        self.clip_gap = 50
        self.video_path = video_path
        self.mask_path = mask_path
        self.video_out_path = os.path.join(
            os.path.dirname(os.path.abspath(self.video_path)),
            f"{os.path.basename(self.video_path).rsplit('.', 1)[0]}_no_sub.mp4"
        )

    def read_frame_info_from_video(self):
        reader = cv2.VideoCapture(self.video_path)
        frame_info = {
            'W_ori': int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),
            'H_ori': int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5),
            'fps': reader.get(cv2.CAP_PROP_FPS),
            'len': int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        }
        return reader, frame_info

    def __call__(self, input_mask=None, input_sub_remover=None, tbar=None):
        reader, frame_info = self.read_frame_info_from_video()
        if input_sub_remover is not None:
            writer = input_sub_remover.video_writer
        else:
            writer = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(
                *"mp4v"), frame_info['fps'], (frame_info['W_ori'], frame_info['H_ori']))
        rec_time = frame_info['len'] // self.clip_gap if frame_info['len'] % self.clip_gap == 0 else frame_info['len'] // self.clip_gap + 1
        split_h = int(frame_info['W_ori'] * 3 / 16)
        if input_mask is None:
            mask = self.sttn_inpaint.read_mask(self.mask_path)
        else:
            _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
            mask = mask[:, :, None]
        inpaint_area = self.sttn_inpaint.get_inpaint_area_by_mask(
            frame_info['H_ori'], split_h, mask)
        for i in range(rec_time):
            start_f = i * self.clip_gap
            end_f = min((i + 1) * self.clip_gap, frame_info['len'])
            print(time.time(), 'Processing:', start_f + 1,
                  '-', end_f, ' / Total:', frame_info['len'])
            frames_hr = []
            frames = {}
            comps = {}
            for k in range(len(inpaint_area)):
                frames[k] = []
            for j in range(start_f, end_f):
                success, image = reader.read()
                frames_hr.append(image)
                for k in range(len(inpaint_area)):
                    image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                    image_resize = cv2.resize(
                        image_crop, (self.sttn_inpaint.model_input_width, self.sttn_inpaint.model_input_height))
                    frames[k].append(image_resize)
            for k in range(len(inpaint_area)):
                comps[k] = self.sttn_inpaint.inpaint(frames[k])
            if inpaint_area:
                for j in range(end_f - start_f):
                    if input_sub_remover is not None and input_sub_remover.gui_mode:
                        original_frame = copy.deepcopy(frames_hr[j])
                    else:
                        original_frame = None
                    frame = frames_hr[j]
                    for k in range(len(inpaint_area)):
                        comp = cv2.resize(
                            comps[k][j], (frame_info['W_ori'], split_h))
                        comp = cv2.cvtColor(np.array(comp).astype(
                            np.uint8), cv2.COLOR_BGR2RGB)
                        mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]
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
        writer.release()