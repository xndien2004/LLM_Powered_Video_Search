import os
import sys
import subprocess
from tqdm import tqdm

def convert_video_to_audio_ffmpeg(video_file, save_path, output_ext="wav"):
    """Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module"""
    subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{save_path}.{output_ext}"], 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    

def main():
    video_dir = "Data/DataSampleAIC23/AIC_Video"
    all_video_paths = dict()
    
    for part in sorted(os.listdir(video_dir)):
        data_path = part.split("_")[-1]
        all_video_paths[data_path] = dict()

    for data_part in sorted(all_video_paths.keys()):
        data_part_path = f'{video_dir}/Videos_{data_part}/video'
        video_paths = sorted(os.listdir(data_part_path))
        video_ids = [video_path.replace('.mp4', '').split('_')[-1] for video_path in video_paths]
        for video_id, video_path in zip(video_ids, video_paths):
            video_path_full = f'{data_part_path}/{video_path}'
            all_video_paths[data_part][video_id] = video_path_full

    save_dir_all = 'media/audio/audio_extracted'

    if not os.path.exists(save_dir_all):
        os.mkdir(save_dir_all)

    for key in tqdm(all_video_paths.keys()):
        save_dir = f'{save_dir_all}/{key}'

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
        video_paths_dict = all_video_paths[key]
        video_ids = sorted(video_paths_dict.keys())
        for video_id in tqdm(video_ids):
            video_path = video_paths_dict[video_id]
            save_path = f'{save_dir}/{video_id}'
            convert_video_to_audio_ffmpeg(video_path, save_path)

    print("Converting Videos to Audio Completed")

if __name__ == "__main__":
    main()