import multiprocessing
from subtitle_remover import SubtitleRemover

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    video_path = input(f"Please input video or image file path: ").strip()
    sub_area = (700, 1200, 20, 700)
    segments_file = 'segments.json'
    SubtitleRemover(video_path, sub_area=sub_area, segments_file=segments_file).run()