import sys
import os
import time
import numpy as np
import scipy
import pickle
import json
import re
from utils import combine_search
from AIC.settings import MEDIA_ROOT
from utils.combine_search import maximal_marginal_relevance

class CaptionRetrieval():
    def __init__(self, id2img_fps, pkl_caption_path, npz_caption_path):
        tfids_caption_path = MEDIA_ROOT+'/contexts_bin/'
        self.tfidf_transform = None
        self.context_matrix = None
        with open(tfids_caption_path + pkl_caption_path, 'rb') as f:
            self.tfidf_transform = pickle.load(f)
        self.context_matrix = scipy.sparse.load_npz(tfids_caption_path + npz_caption_path)
    
        self.id2img_fps = id2img_fps
    
    def load_json_file(self, json_file):
        with open(json_file, 'r') as f:
            js = json.load(f)
        return {int(k):v for k,v in js.items()}
    
    def transform_input(self, input_query:str):
        '''
        This function transform input take from user to tf-idf array
        It remove all word not in the vocabulary/corpus
        
        input:
        input: a string text used as query take from user
        
        output:
        numpy array converted from query with tf-idf
        '''
        vectorize = self.tfidf_transform.transform([input_query])
        return vectorize
    
    def __call__(self, texts, is_mmr=False, lambda_param=0.5, k=100, index=None,):
        k = k*2 if is_mmr else k
        scores, idx_image_ = self.find_similar_score(texts, k, index=index)
        if is_mmr:
            print("use MMR")
            input_vector = self.transform_input(texts)
            selected_indices = maximal_marginal_relevance(input_vector, self.context_matrix[idx_image_,:], lambda_param=lambda_param, top_k=k)
            idx_image_ = np.array(idx_image_)[selected_indices]
            scores = scores[selected_indices]
        infos_query = list(map(self.id2img_fps.get, list(idx_image_)))
        image_paths = [info['image_path'] for info in infos_query]
        frame_idx = [info['pts_time'] for info in infos_query]
        return scores, idx_image_, frame_idx, image_paths
    
    def find_similar_score(
            self,
            text:str,
            k:int,
            index,
    ):
        vectorize = self.transform_input(text,)
        if index is None: 
            scores = vectorize.dot(self.context_matrix.T).toarray()[0]
            sort_index = np.argsort(scores)[::-1][:k]
            scores = scores[sort_index]
        else:
            scores = vectorize.dot(self.context_matrix[index,:].T).toarray()[0]
            sort_index = np.argsort(scores)[::-1][:k]
            scores = scores[sort_index]
            sort_index = np.array(index)[sort_index]
        return scores, sort_index

