import json
import torch
import clip
import faiss
import numpy as np
import os
from PIL import Image
import gc
import open_clip

from .nlp_processing import translate_lib
from .video_retrieval import asr_retrieval, caption_retrieval, ocr_retrieval, tag_retrieval, object_retrieval
from .combine_search import maximal_marginal_relevance

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class FaissSearch:
    def __init__(self, dict_path: dict, is_openclip=False, is_object=False, is_evalip=False):     
        self.id2img_fps = self.load_json_file(dict_path['id2img_fps_json_path'])
        self.map_asr = self.load_json_file(dict_path['map_asr_json_path'])
        self.caption_retrieval = caption_retrieval.CaptionRetrieval(self.id2img_fps, dict_path['pkl_caption_path'], dict_path['npz_caption_path'])
        self.ocr_retrieval = ocr_retrieval.OcrRetrieval(self.id2img_fps, dict_path['dict_pkl_ocr_path'], dict_path['dict_npz_ocr_path'])
        self.asr_retrieval = asr_retrieval.ASRRetrieval(self.map_asr, dict_path['dict_pkl_asr_path'], dict_path['dict_npz_asr_path'])
        self.tag_retrieval = tag_retrieval.TagRetrieval(self.id2img_fps, dict_path['dict_pkl_tag_path'], dict_path['dict_npz_tag_path'])
        self.__device = "cuda" if torch.cuda.is_available() else "cpu" 
        
        self.is_openclip = is_openclip
        if is_object:
            self.object_retrieval = object_retrieval.ObjectRetrieval(self.id2img_fps, dict_path['dict_pkl_object_path'], dict_path['dict_npz_object_path'])
        if is_openclip:
            self.index_open_clip = self.load_bin_file(dict_path['faiss_openclip_bin_path'])
            # self.open_clip_model, _, self.openclip_preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP-384', device=self.__device, pretrained='webli')
            # self.open_clip_tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP-384')
            self.open_clip_model, _, self.openclip_preprocess = open_clip.create_model_and_transforms('ViT-L-14', device=self.__device, pretrained='datacomp_xl_s13b_b90k')
            self.open_clip_tokenizer = open_clip.get_tokenizer('ViT-L-14')
        if is_evalip:
            self.index_evalip = self.load_bin_file(dict_path['faiss_evalip_bin_path'])
            self.evalip_model, _, self.evalip_preprocess = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', device=self.__device, pretrained='dfn5b')
            self.evalip_tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')


    def load_json_file(self, json_path: str): 
        with open(json_path, 'r') as f: 
            js = json.load(f)
        return {int(k): v for k, v in js.items()}
    
    def load_txt_file(self, txt_path: str):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        return lines
    
    def load_bin_file(self, bin_file: str): 
        return faiss.read_index(bin_file)

    def get_frame_info(self, idx_image):
        infos_query = list(map(self.id2img_fps.get, list(idx_image)))
        image_paths = [info['image_path'] for info in infos_query]
        frame_idx = [info['pts_time'] for info in infos_query]
        return frame_idx, image_paths
    
    def get_frame_info_single(self, idx_image):
        info_query = self.id2img_fps.get(idx_image)
        image_path = info_query['image_path']
        frame_idx = info_query['pts_time']
        return frame_idx, image_path

    
    def text_search_openclip(self, text: str, index=None, is_mmr=False, lambda_param=0.5, k=5, is_translate=True):
        # translate
        if is_translate:
            text = str(translate_lib(text, to_lang='en'))

        ###### TEXT FEATURES EXTRACTING ######
        text = self.open_clip_tokenizer([text]).to(self.__device)
        text_features = self.open_clip_model.encode_text(text)

        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().detach().numpy().astype(np.float32)

        ###### SEARCHING #####
        index_choosed = self.index_open_clip
        k = k*2 if is_mmr else k

        if index is None:
            scores, idx_image = index_choosed.search(text_features, k=k)
        else:
            id_selector = faiss.IDSelectorArray(index)
            scores, idx_image = index_choosed.search(text_features, k=k, 
                                                     params=faiss.SearchParametersIVF(sel=id_selector))
        idx_image = idx_image.flatten() 
        scores = scores.flatten()

        if is_mmr:
            print("use MMR")
            doc_embeddings = [self.index_open_clip.reconstruct(int(idx)) for idx in idx_image]
            selected_indices = maximal_marginal_relevance(text_features[0], doc_embeddings, lambda_param=lambda_param, top_k=k)
            idx_image = np.array(idx_image)[selected_indices]
            scores = scores[selected_indices]
        ###### GET INFOS KEYFRAMES_ID ######
        frame_idx, image_paths = self.get_frame_info(idx_image)
        
        return scores, idx_image, frame_idx , image_paths

    def text_search_evalip(self, text: str, index=None, is_mmr=False, lambda_param=0.5, k=5, is_translate=True):
        # translate
        if is_translate:
            text = str(translate_lib(text, to_lang='en'))

        ###### TEXT FEATURES EXTRACTING ######
        text = self.evalip_tokenizer([text]).to(self.__device)
        text_features = self.evalip_model.encode_text(text)

        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().detach().numpy().astype(np.float32)

        ###### SEARCHING #####
        index_choosed = self.index_evalip
        k = k*2 if is_mmr else k

        if index is None:
            scores, idx_image = index_choosed.search(text_features, k=k)
        else:
            id_selector = faiss.IDSelectorArray(index)
            scores, idx_image = index_choosed.search(text_features, k=k, 
                                                     params=faiss.SearchParametersIVF(sel=id_selector))
        idx_image = idx_image.flatten() 
        scores = scores.flatten()

        if is_mmr:
            print("use MMR")
            doc_embeddings = [self.index_evalip.reconstruct(int(idx)) for idx in idx_image]
            selected_indices = maximal_marginal_relevance(text_features[0], doc_embeddings, lambda_param=lambda_param, top_k=k)
            idx_image = np.array(idx_image)[selected_indices]
            scores = scores[selected_indices]

        ###### GET INFOS KEYFRAMES_ID ######
        frame_idx, image_paths = self.get_frame_info(idx_image)
        return scores, idx_image, frame_idx , image_paths
    
    def image_search(self, image_path, is_mmr=False, lambda_param=0.5, k=5, index=None):
        image = Image.open(image_path).convert("RGB")
        image_input = (self.evalip_preprocess(image) if not self.is_openclip 
                    else self.openclip_preprocess(image)).to(self.__device).unsqueeze(0)
        
        # Encode image features
        model = self.evalip_model if not self.is_openclip else self.open_clip_model
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        
        # Normalize image features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().detach().numpy()
        k = k*2 if is_mmr else k

        # Select FAISS index based on the model
        index_search = self.index_evalip if not self.is_openclip else self.index_open_clip
        print("EVA-Clip model is used" if not self.is_openclip else "")
        
        # Perform search with FAISS
        if index is None:
            scores, idx_image = index_search.search(image_features, k)
        else:
            id_selector = faiss.IDSelectorArray(index)
            search_params = faiss.SearchParametersIVF(sel=id_selector)
            scores, idx_image = index_search.search(image_features, k=k, params=search_params)

        # Flatten and get additional info
        idx_image = idx_image.flatten()
        scores = scores.flatten()

        if is_mmr:
            print("use MMR")
            doc_embeddings = [index_search.reconstruct(int(idx)) for idx in idx_image]
            selected_indices = maximal_marginal_relevance(image_features[0], doc_embeddings, lambda_param=lambda_param, top_k=k)
            idx_image = np.array(idx_image)[selected_indices]
            scores = scores[selected_indices]

        frame_idx, image_paths = self.get_frame_info(idx_image)
        return scores, idx_image, frame_idx, image_paths

    
    def context_search(self, object_input, is_mmr=False, lambda_param=0.5, k=5 , index=None):
        '''
        Example:
        inputs = {
            'bbox': "a0person",
            'number': "person0, person1",
            'color':None,
            'tag_bbox':None,
            'number_tag': "person0 person1",
        }
        ''' 
        if object_input is not None:
            scores, idx_image, frame_idx, image_paths = self.object_retrieval(object_input, is_mmr=is_mmr, lambda_param=lambda_param, k=k, index=index)
        return scores, idx_image, frame_idx, image_paths
    
    def caption_search(self, texts, is_mmr=False, lambda_param=0.5, k=5, index=None, is_translate=True):
        '''
        Example:
        texts = "Hình ảnh là cảnh trong bản tin HTV7 HD. Người dẫn chương trình mặc áo xanh và cà vạt chấm bi, đứng trước nền thành phố lúc hoàng hôn. Dưới cùng có dải chữ chạy thông tin sự kiện."
        '''
        if is_translate:
            texts = str(translate_lib(texts, to_lang='en'))
        if texts != '':
            scores, idx_image, frame_idx, image_paths = self.caption_retrieval(texts, is_mmr=is_mmr, lambda_param=lambda_param, k=k, index=index)
        return scores, idx_image, frame_idx, image_paths
    
    def ocr_search(self, texts, is_mmr=False, lambda_param=0.5, k=5, index=None):
        '''
        Example:
        texts = "Hình ảnh là cảnh trong bản tin HTV7 HD. Người dẫn chương trình mặc áo xanh và cà vạt chấm bi, đứng trước nền thành phố lúc hoàng hôn. Dưới cùng có dải chữ chạy thông tin sự kiện."
        '''
        if texts != '':
            scores, idx_image, frame_idx, image_paths = self.ocr_retrieval(texts, is_mmr=is_mmr, lambda_param=lambda_param, k=k, index=index)
        return scores, idx_image, frame_idx, image_paths

    def asr_search(self, texts, is_mmr=False, lambda_param=0.5, k=5, index=None):
        '''
        Example:
        texts = "Hình ảnh là cảnh trong bản tin HTV7 HD. Người dẫn chương trình mặc áo xanh và cà vạt chấm bi, đứng trước nền thành phố lúc hoàng hôn. Dưới cùng có dải chữ chạy thông tin sự kiện."
        '''
        if texts != '':
            scores, idx_image, frame_idx, image_paths = self.asr_retrieval(texts, is_mmr=is_mmr, lambda_param=lambda_param, k=k, index=index)
        return scores, idx_image, frame_idx, image_paths
    
    def tag_search(self, texts, is_mmr=False, lambda_param=0.5, k=5, index=None):
        '''
        Example:
        texts = "building sky tree"
        '''
        if texts != '':
            scores, idx_image, frame_idx, image_paths = self.tag_retrieval(texts, is_mmr=is_mmr, lambda_param=lambda_param, k=k, index=index)
        return scores, idx_image, frame_idx, image_paths
        
    def feelback(self, previous_results, positive_feedback_idxs, negative_feedback_idxs):
        
        feedback_idxs = np.array(positive_feedback_idxs + negative_feedback_idxs, dtype='int64')
        num_positive_feedbacks = len(positive_feedback_idxs)

        video_score_map = {video_id: score 
                        for result in previous_results
                        for video_id, score in zip(result['video_info']['lst_idxs'], result['video_info']['lst_scores'])}

        for video_id in negative_feedback_idxs:
            video_score_map.pop(video_id, None)

        valid_video_ids = np.array(list(video_score_map.keys()), dtype='int64')

        index_search = self.index_evalip if not self.is_openclip else self.index_open_clip
        feedback_features = index_search.reconstruct_batch(feedback_idxs)

        video_selector = faiss.IDSelectorArray(valid_video_ids)
        search_params = faiss.SearchParametersIVF(sel=video_selector)

        feelback_scores, feelback_idx_images = index_search.search(
            feedback_features, 
            k=len(valid_video_ids), 
            params=search_params
        )

        for idx, (scores, idx_images) in enumerate(zip(feelback_scores, feelback_idx_images)):
            is_positive_feedback = idx < num_positive_feedbacks
            score_adjustment = 1 if is_positive_feedback else -1

            for score, video_id in zip(scores, idx_images):
                video_score_map[video_id] += score * score_adjustment

        feelbacked_results = sorted(video_score_map.items(), key=lambda x: x[1], reverse=True)
        feelbacked_ids = [video_id for video_id, _ in feelbacked_results]
        feelbacked_scores = [score for _, score in feelbacked_results]

        video_info_list = [self.id2img_fps.get(video_id) for video_id in feelbacked_ids]
        feelbacked_framesID = [info['pts_time'] for info in video_info_list if info is not None]
        image_paths = [info['image_path'] for info in video_info_list if info is not None]

        return feelbacked_scores, feelbacked_ids, feelbacked_framesID, image_paths


def main():
    bin_file = 'media/faiss_clip.bin'
    json_path = 'media/id2img_fps.json'

    cosine_faiss = FaissSearch(bin_file, json_path)

    # Example 1: Text-based search
    text = 'Hình ảnh là cảnh trong bản tin HTV7 HD. Người dẫn chương trình mặc áo xanh và cà vạt chấm bi, đứng trước nền thành phố lúc hoàng hôn. Dưới cùng có dải chữ chạy thông tin sự kiện.'
    scores, _, frame_idx, image_paths = cosine_faiss.text_search(text, None, k=10)
    
    print("--- Text Search ---")
    print("Scores:", scores)
    print("Frame indices:", frame_idx)
    print("Image paths:", image_paths)

    # Example 2: Image-based search
    image_path = r"D:\HocTap\CuocThi\Backup\keyframes\L01_V002\0039.jpg"
    scores, _, frame_idx, image_paths = cosine_faiss.image_search(image_path, k=10)
    
    print("--- Image Search ---")
    print("Scores:", scores)
    print("Frame indices:", frame_idx)
    print("Image paths:", image_paths)

if __name__ == "__main__":
    main()
