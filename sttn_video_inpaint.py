import cv2
import os
import numpy as np
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

    def __call__(self, frames, input_mask=None, input_sub_remover=None, tbar=None):
        if input_sub_remover is not None:
            writer = input_sub_remover.video_writer
        else:
            reader = cv2.VideoCapture(self.video_path)
            fps = reader.get(cv2.CAP_PROP_FPS)
            size = (int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            writer = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(
                *"mp4v"), fps, size)

        split_h = int(frames[0].shape[1] * 3 / 16)
        if input_mask is None:
            mask = self.sttn_inpaint.read_mask(self.mask_path)
        else:
            _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
            mask = mask[:, :, None]
        inpaint_area = self.sttn_inpaint.get_inpaint_area_by_mask(
            frames[0].shape[0], split_h, mask)

        frames_hr = frames
        frames_scaled = {}
        comps = {}
        for k in range(len(inpaint_area)):
            frames_scaled[k] = []

        for j in range(len(frames_hr)):
            image = frames_hr[j]
            for k in range(len(inpaint_area)):
                image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                image_resize = cv2.resize(
                    image_crop, (self.sttn_inpaint.model_input_width, self.sttn_inpaint.model_input_height))
                frames_scaled[k].append(image_resize)

        for k in range(len(inpaint_area)):
            comps[k] = self.sttn_inpaint.inpaint(frames_scaled[k])

        if inpaint_area:
            for j in range(len(frames_hr)):
                frame = frames_hr[j]
                for k in range(len(inpaint_area)):
                    comp = cv2.resize(
                        comps[k][j], (frame.shape[1], split_h))
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
        if input_sub_remover is None:
            writer.release()
        return frames_hr