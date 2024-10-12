import os
import json
import zipfile
import kenlm
import torch
import librosa    
from tqdm import tqdm
import soundfile as sf
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel


# Get model
def get_decoder_ngram_model(tokenizer, ngram_lm_path):
    vocab_dict = tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab = [x[1] for x in sort_vocab][:-2]
    vocab_list = vocab
    # convert ctc blank character representation
    vocab_list[tokenizer.pad_token_id] = ""
    # replace special characters
    vocab_list[tokenizer.unk_token_id] = ""
    # vocab_list[tokenizer.bos_token_id] = ""
    # vocab_list[tokenizer.eos_token_id] = ""
    # convert space character representation
    vocab_list[tokenizer.word_delimiter_token_id] = " "
    # specify ctc blank char index, since conventially it is the last entry of the logit matrix
    alphabet = Alphabet.build_alphabet(vocab_list, ctc_token_idx=tokenizer.pad_token_id)
    lm_model = kenlm.Model(ngram_lm_path)
    decoder = BeamSearchDecoderCTC(alphabet,
                                   language_model=LanguageModel(lm_model))
    return decoder


def main():
    # load model and tokenizer
    processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
    model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
    lm_file = hf_hub_download("nguyenvulebinh/wav2vec2-base-vietnamese-250h", filename='vi_lm_4grams.bin.zip')
    with zipfile.ZipFile(lm_file, 'r') as zip_ref:
        zip_ref.extractall('./Data')
    ngram_lm_model = get_decoder_ngram_model(processor.tokenizer, 'Data/vi_lm_4grams.bin')

    # Parse audio information
    audios_dir = 'media/audio/audio_extracted'
    all_audio_paths = dict()
    for part in sorted(os.listdir(audios_dir)):
        all_audio_paths[part] =  dict()

    for data_part in sorted(all_audio_paths.keys()):
        data_part_path = f'{audios_dir}/{data_part}'
        audio_paths = sorted(os.listdir(data_part_path))
        for audio_path in audio_paths:
            audio_id = audio_path.replace('.wav', '')
            audio_path_full = f'{data_part_path}/{audio_path}'
            all_audio_paths[data_part][audio_id] = audio_path_full

    # Audio recognition
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    save_dir_all = 'media/audio/audio_recognized'
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
            
            speech, sampling_rate = librosa.load(audio_path, mono=True, sr=16000)
            speech = speech.astype('float64')
            speech_len = len(speech)
            
            with open(f'media/audio/audio_detected/{key}/{audio_id}.json', 'r') as f:
                audio_shots = json.load(f)
            
            results = []
            for audio_shot in audio_shots:
                start, end = audio_shot
                lst_audio_frames = []
                while (end-start) >= 1:
                    if (end-start) <= 10:
                        lst_audio_frames.append(speech[int(start*sampling_rate):min(speech_len, round(end*sampling_rate))])
                        break
                    else:
                        lst_audio_frames.append(speech[int(start*sampling_rate):min(speech_len, round((start+10)*sampling_rate))])
                        start = start+10
                if lst_audio_frames != []:
                    input_values = processor(lst_audio_frames, sampling_rate=sampling_rate, return_tensors="pt", padding="longest").input_values.to(device)
                    logits = model(input_values).logits
                    result = []
                    for logit in logits:
                        beam_search_output = ngram_lm_model.decode(logit.cpu().detach().numpy(), beam_width=500)
                        result.append(beam_search_output)
                    result = " ".join(result)
                    results.append(result)
                else:
                    results.append("")

            with open(f'{save_dir}/{audio_id}.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False)

if __name__ == "__main__":
    main()