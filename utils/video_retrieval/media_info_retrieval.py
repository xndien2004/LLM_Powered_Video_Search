import os
import sys
import glob
import scipy
import pickle
import numpy as np
import json
import torch
import io
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import combine_search
from django.conf import settings
from AIC.settings import MEDIA_ROOT

# class CPU_Unpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name == '_load_from_bytes':
#             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#         else: return super().find_class(module, name)

class MediaInfoRetrieval():
    def __init__(self, id2img_fps, dict_pkl_media_info_path, dict_npz_media_info_path):
        tfids_media_info_path = MEDIA_ROOT+'/contexts_bin/'
        self.all_datatype = ['description', 'title']
        self.tfidf_transform = {}
        self.context_matrix = {}
        for data_type in self.all_datatype:
            with open(tfids_media_info_path + dict_pkl_media_info_path[data_type], 'rb') as f:
                self.tfidf_transform[data_type] = pickle.load(f)
            self.context_matrix[data_type] = scipy.sparse.load_npz(tfids_media_info_path + dict_npz_media_info_path[data_type])

        self.id2img_fps = self.load_json_file(id2img_fps)

    def transform_input(self, input_query:str, transform_type:str,):
        '''
        This function transform input take from user to tf-idf array
        It remove all word not in the vocabulary/corpus
        
        input:
        input: a string text used as query take from user
        
        output:
        numpy array converted from query with tf-idf
        '''
        if transform_type in self.all_datatype:
            vectorize = self.tfidf_transform[transform_type].transform([input_query])
        else:
            print('this type does not support')
            sys.exit()
        return vectorize

    def __call__(self, texts, k=100, index=None,sources=None):
        list_results = []
        for input_type in self.all_datatype:
            if texts[input_type] != '':
                scores_, idx_video = self.find_similar_score(text=texts[input_type], transform_type=input_type, k=k, index=index)
                infos_query = list(map(self.id2img_fps.get, list(idx_video)))
                video_paths = [info['video_path'] for info in infos_query]
                watch_urls = [info['watch_url'] for info in infos_query]
                list_results.append((scores_, idx_video, watch_urls, video_paths, list(range(1,len(watch_urls)+1)),[sources]*len(watch_urls)))
        
        scores, idx_video, watch_urls, video_paths,_ = combine_search.combined_ranking_score(list_results, alpha=0.5, beta=0.5)
        return scores, idx_video, watch_urls, video_paths  


    def find_similar_score(
            self,
            text:str,
            transform_type:str,
            k:int,
            index,
    ):
        vectorize = self.transform_input(text, transform_type)
        if index is None: 
            scores = vectorize.dot(self.context_matrix[transform_type].T).toarray()[0]
            sort_index = np.argsort(scores)[::-1][:k]
            scores = scores[sort_index]
        else:
            scores = vectorize.dot(self.context_matrix[transform_type][index,:].T).toarray()[0]
            sort_index = np.argsort(scores)[::-1][:k]
            scores = scores[sort_index]
            sort_index = np.array(index)[sort_index]
        return scores, sort_index

    def load_json_file(self, json_path: str):
        with open(json_path, 'r') as f: 
            js = json.load(f)
        return {int(k):v for k,v in js.items()}