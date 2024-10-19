import tempfile
import os
import time
import glob
import json
from collections import Counter
from AIC.settings import MEDIA_ROOT, STATICFILES_DIRS
import numpy as np
import pandas as pd

def get_n_cluster_frames(n: int, element: str, list_elements: list): 
    index = list_elements.index(element) 
    n_elements = list_elements[max(0, index - n):min(len(list_elements), index + n + 1)]
    return n_elements


def save_image(image_file, format='.png'):
    with tempfile.NamedTemporaryFile(suffix=format, delete=False) as temp_file:
        for chunk in image_file.chunks():
            temp_file.write(chunk) 
    return  temp_file.name  

def get_video_info(video_id, id_frame):
    media_info_path = str(MEDIA_ROOT + "/media-info/")
    media_info_json = glob.glob(media_info_path + video_id + ".json")
    if not media_info_json:
        return None
    with open(media_info_json[0], 'r', encoding='utf-8', errors='ignore') as f:
        data = json.load(f)

    map_keyframes_path = str(MEDIA_ROOT + "/map-keyframes/")
    map_keyframes_csv = glob.glob(map_keyframes_path + video_id + ".csv")
    if not map_keyframes_csv:
        return None
    df = pd.read_csv(map_keyframes_csv[0])
    df = df[df['n'] == int(id_frame)]
    start_time = df['fps'].values[0]
    return data, int(start_time)


def extract_video_id(path):
    if path != '':
        return path.split('/')[-2]
    return ''

def sort_by_frequent_video(scores, idx_image, frame_idx, image_paths):
    video_ids = [extract_video_id(path) for path in image_paths]
    video_count = Counter(video_ids)
    
    sorted_video_ids = [video_id for video_id, _ in sorted(video_count.items(), key=lambda x: x[1], reverse=True)]
    
    combined = list(zip(scores, idx_image, frame_idx, image_paths, video_ids))
    result = []
    for video_id in sorted_video_ids:
        filtered_items = [item for item in combined if item[4] == video_id]
        
        if filtered_items:
            scores, idx_image, frame_idx, image_paths, _ = zip(*filtered_items)
            result.append([video_id, list(scores), list(idx_image), list(frame_idx), list(image_paths)])
    
    return result

def get_extra_noextra(idx_image, source):
    if not source or not idx_image:
        return [], []
    extra_index = [idx_image[i] for i in range(len(source)) if source[i] == "extra"]
    noextra_index = [idx_image[i] for i in range(len(source)) if source[i] == "no_extra"]
    return extra_index, noextra_index

def search_by_type(search_type, search_func, source, text, is_mmr, lambda_param, index=None, k=None, list_results=None):
    scores, idx_image, frame_idxs, image_paths = search_func(text, is_mmr=is_mmr, lambda_param=lambda_param, index=index, k=k)
    list_results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), [source]*len(frame_idxs)))
    print(f"{search_type} search complete")

def run_search(choice_key, search_func, source, choice_dict, text, is_mmr, lambda_param, index, k, list_results):
    if choice_dict.get(choice_key):
        search_by_type(choice_key.upper(), search_func, source=source, text=text, is_mmr=is_mmr, lambda_param=lambda_param, index=index, k=k, list_results=list_results)

def get_index_image(json_data, keyframe):
    keyframes = keyframe.split(",") if isinstance(keyframe, str) and "," in keyframe else [keyframe]
    return [int(key) for key, value in json_data.items() 
            for kf in keyframes if kf in value["image_path"]]
