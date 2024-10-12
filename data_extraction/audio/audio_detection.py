import os
import torch
import glob
import json
from tqdm import tqdm
from pyannote.audio import Pipeline

def main():
    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", 
                                        use_auth_token="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")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline=pipeline.to(device)

    # Parse audio information
    audios_dir = "media/audio/audio_extracted"
    all_audio_paths = dict()
    for part in sorted(os.listdir(audios_dir)):
        if part == '.DS_Store':
            continue
        all_audio_paths[part] =  dict()

    for data_part in sorted(all_audio_paths.keys()):
        data_part_path = f'{audios_dir}/{data_part}'
        audio_paths = sorted(os.listdir(data_part_path))
        for audio_path in audio_paths:
            audio_id = audio_path.replace('.wav', '')
            audio_path_full = f'{data_part_path}/{audio_path}'
            all_audio_paths[data_part][audio_id] = audio_path_full

    # Audio detection
    save_dir_all = 'media/audio/audio_detected'
    if not os.path.exists(save_dir_all):
        os.mkdir(save_dir_all)

    for key in tqdm(all_audio_paths.keys()):
        save_dir = f'{save_dir_all}/{key}'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    audio_paths_dict = all_audio_paths[key]
    audio_ids = sorted(audio_paths_dict.keys())
    for audio_id in tqdm(audio_ids):
        audio_path = audio_paths_dict[audio_id]
        output = pipeline(audio_path)
        
        result = []
        for speech in output.get_timeline().support():
            result.append([speech.start, speech.end])
            
        with open(f'{save_dir}/{audio_id}.json', 'w') as f:
            json.dump(result, f)

if __name__ == "__main__":
    main()