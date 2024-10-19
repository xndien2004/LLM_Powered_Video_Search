import os
import sys
import glob
import scipy
import pickle
import numpy as np
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import combine_search
from AIC.settings import MEDIA_ROOT
from utils.combine_search import maximal_marginal_relevance

class ObjectRetrieval():
    def __init__(self, id2img_fps, dict_pkl_object_path, dict_npz_object_path):
        tfids_object_path = MEDIA_ROOT+'/contexts_bin/' #'bbox', 'color','tag_bbox',
        self.all_datatype = ['number', 'number_tag','synthetic']
        self.tfidf_transform = {}
        self.context_matrix = {}
        for data_type in self.all_datatype:
            with open(tfids_object_path + dict_pkl_object_path[data_type], 'rb') as f:
                self.tfidf_transform[data_type] = pickle.load(f)
            self.context_matrix[data_type] = scipy.sparse.load_npz(tfids_object_path + dict_npz_object_path[data_type])

        self.id2img_fps = id2img_fps

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

    def __call__(self, texts, is_mmr=False, lambda_param=0.5, k=100, index=None,sources=None):
        list_results = []
        for input_type in self.all_datatype:
            if texts[input_type] != '':
                k = k*2 if is_mmr else k
                scores_, idx_image_ = self.find_similar_score(texts[input_type], input_type, k, index=index)
                if is_mmr:
                    input_vector = self.transform_input(texts[input_type], input_type)
                    selected_indices = maximal_marginal_relevance(input_vector, self.context_matrix[input_type][idx_image_,:], lambda_param=lambda_param, top_k=k)
                    idx_image_ = np.array(idx_image_)[selected_indices]
                    scores_ = scores_[selected_indices]
                infos_query = list(map(self.id2img_fps.get, list(idx_image_)))
                image_paths = [info['image_path'] for info in infos_query]
                frame_idx = [info['pts_time'] for info in infos_query]
                list_results.append((scores_, idx_image_, frame_idx, image_paths, list(range(1,len(frame_idx)+1)),[sources]*len(frame_idx)))
        
        scores, idx_image, frame_idx, image_paths,_ = combine_search.combined_ranking_score(list_results, alpha=0.5, beta=0.5)
        return scores, idx_image, frame_idx, image_paths  


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