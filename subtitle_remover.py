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
        
        # 创建临时目录而不是单个文件
        self.temp_dir = tempfile.mkdtemp()
        self.video_temp_file = os.path.join(self.temp_dir, 'temp_video.mp4')
        self.video_writer = cv2.VideoWriter(
            self.video_temp_file, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
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
        frames_processed = 0
        while True:
            success, frame = self.video_cap.read()
            if not success:
                break
            if frame_index in self.segment_frames:
                # 处理需要修复的帧
                sttn_video_inpaint([frame], input_mask=mask, input_sub_remover=self, tbar=tbar)
                frames_processed += 1
            else:
                # 不需要修复的帧直接写入
                self.video_writer.write(frame)
                if tbar is not None:
                    self.update_progress(tbar, increment=1)
            frame_index += 1
        
        print(f"Total frames processed: {frames_processed}")
        print(f"Total frames written: {frame_index}")

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
        
        # 检查临时视频文件是否存在
        if os.path.exists(self.video_temp_file):
            print(f"Temp video file size: {os.path.getsize(self.video_temp_file) / (1024*1024):.2f} MB")
        else:
            print(f"Temp video file {self.video_temp_file} does not exist!")
        
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
        
        # 清理临时目录
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Temp directory {self.temp_dir} deleted successfully")
            except Exception as e:
                print(f"Failed to delete temp directory: {e}")

    def merge_audio_to_video(self):
        temp_dir = self.temp_dir  # 使用已创建的临时目录
        video_temp_path = self.video_temp_file
        audio_temp_path = os.path.join(temp_dir, "temp_audio")  # 不指定后缀，自动适配编码

        # 1. 提取音频（通用处理，支持多种编码）
        print("Extracting audio...")
        audio_extract_command = [
            "ffmpeg", "-y", "-i", self.video_path,
            "-vn", "-c:a", "copy",  # 直接复制音频流，不重新编码
            "-loglevel", "error", audio_temp_path
        ]
        use_shell = platform.system() == "Windows"

        audio_extracted = False
        try:
            subprocess.run(audio_extract_command, check=True, shell=use_shell)
            audio_extracted = os.path.exists(audio_temp_path) and os.path.getsize(audio_temp_path) > 0
            if audio_extracted:
                print(f"Audio extracted successfully ({os.path.getsize(audio_temp_path)/1024:.2f} KB)")
        except Exception as e:
            print(f"Audio extraction failed: {str(e)}")

        # 2. 合并视频和音频（支持有/无音频两种情况）
        if audio_extracted:
            print("Merging audio and video...")
            output_ext = ".mp4"  # 输出为MP4格式
            merge_command = [
                "ffmpeg", "-y", "-i", video_temp_path,
                "-i", audio_temp_path,
                "-vcodec", "libx264", "-acodec", "copy",  # 视频重新编码为H.264，音频直接复制
                "-loglevel", "error", self.video_out_name
            ]
            try:
                subprocess.run(merge_command, check=True, shell=use_shell)
                self.is_successful_merged = True
                print(f"Video generated with audio ({os.path.getsize(self.video_out_name)/1024/1024:.2f} MB)")
            except Exception as e:
                print(f"Merge failed: {str(e)}，尝试生成无音频视频...")
                self.fallback_to_video_only(video_temp_path)
        else:
            print("No audio found or extraction failed，生成无音频视频...")
            self.fallback_to_video_only(video_temp_path)

    def fallback_to_video_only(self, video_temp_path):
        """回退：直接复制临时视频文件（无音频）"""
        try:
            shutil.copy2(video_temp_path, self.video_out_name)
            self.is_successful_merged = True
            print(f"Video generated without audio ({os.path.getsize(self.video_out_name)/1024/1024:.2f} MB)")
        except Exception as e:
            print(f"Fallback failed: {str(e)}")
            self.is_successful_merged = False