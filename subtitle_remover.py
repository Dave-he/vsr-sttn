import cv2
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import json
from sttn_video_inpaint import STTNVideoInpaint
from data_processing import create_mask
from config import SKIP_FRAME_DIFF, DEVICE


class SubtitleRemover:
    def __init__(self, vd_path, sub_area=None, gui_mode=False, segments_file=None):
        self.lock = threading.RLock()
        self.sub_area = sub_area
        self.gui_mode = gui_mode
        self.is_picture = False
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(vd_path)
        self.vd_name = Path(self.video_path).stem
        self.frame_count = int(self.video_cap.get(
            cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.mask_size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_temp_file = tempfile.NamedTemporaryFile(
            suffix='.mp4', delete=False)
        self.video_writer = cv2.VideoWriter(
            self.video_temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
        self.video_out_name = os.path.join(os.path.dirname(
            self.video_path), f'{self.vd_name}_no_sub.mp4')
        self.video_inpaint = None
        self.lama_inpaint = None
        self.ext = os.path.splitext(vd_path)[-1]
     

        print(f'use {DEVICE} for acceleration')
        self.progress_total = 0
        self.progress_remover = 0
        self.isFinished = False
        self.preview_frame = None
        self.is_successful_merged = False
        self.segments = []
        if segments_file:
            try:
                with open(segments_file, 'r') as f:
                    self.segments = json.load(f)
            except FileNotFoundError:
                print(f"Error: The file {segments_file} was not found.")
            except json.JSONDecodeError:
                print(f"Error: Failed to parse the JSON file {segments_file}.")

        self.segment_frames = []
        if self.segments:
            for segment in self.segments:
                start_frame = max(0, int(segment["startTime"] * self.fps - SKIP_FRAME_DIFF))
                end_frame = min(self.frame_count, int(segment["endTime"] * self.fps + SKIP_FRAME_DIFF))
                self.segment_frames.extend(range(start_frame, end_frame))

    @staticmethod
    def get_coordinates(dt_box):
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
        for start_no, end_no in continuous_frame_no_list:
            if start_no == frame_no:
                return True
        return False

    @staticmethod
    def find_frame_no_end(frame_no, continuous_frame_no_list):
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

        frame_index = 0
        while True:
            success, frame = self.video_cap.read()
            if not success:
                break
            if frame_index in self.segment_frames:
                # 处理需要修复的帧
                sttn_video_inpaint([frame], input_mask=mask, input_sub_remover=self, tbar=tbar)
            else:
                # 不需要修复的帧直接写入
                self.video_writer.write(frame)
                if tbar is not None:
                    self.update_progress(tbar, increment=1)
            frame_index += 1

    def run(self):
        start_time = time.time()
        self.progress_total = 0
        if self.segments:
            total_frames_to_process = len(self.segment_frames)
        else:
            total_frames_to_process = self.frame_count
        tbar = tqdm(total=total_frames_to_process, unit='frame', position=0, file=sys.__stdout__,
                    desc='Subtitle Removing')

        self.sttn_mode_with_no_detection(tbar)
        self.video_cap.release()
        self.video_writer.release()
        if not self.is_picture:
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
        if os.path.exists(self.video_temp_file.name):
            audio_merge_command = ["ffmpeg",
                                   "-y", "-i", self.video_temp_file.name,
                                   "-i", temp.name,
                                   "-vcodec", "libx264",
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
        temp.close()
        if not self.is_successful_merged:
            try:
                shutil.copy2(self.video_temp_file.name,
                             self.video_out_name)
            except IOError as e:
                print("Unable to copy file. %s" % e)
        self.video_temp_file.close()