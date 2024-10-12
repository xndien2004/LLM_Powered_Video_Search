import os
import sys
import glob
import scipy
import pickle
import numpy as np
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, load_npz
from scipy import sparse as sp

# script_dir = os.path.dirname(os.path.abspath(__file__))
# print("Script Directory: ", script_dir)
# parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
# print("Parent Directory: ", parent_dir)
# grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
# print("Grand Directory: ", grand_dir)
# sys.path.extend([parent_dir, grand_dir])
# print("System Path: ", sys.path)

def GET_PROJECT_ROOT():
    # goto the root folder of LogBar
    current_abspath = os.path.abspath(__file__)
    print("Current abspath: ", current_abspath)
    while True:
        if os.path.split(current_abspath)[1] == 'VN_Multi_User_Video_Search':
            project_root = current_abspath
            break
        else:
            current_abspath = os.path.dirname(current_abspath)
            # break
    return project_root

PROJECT_ROOT = GET_PROJECT_ROOT()
print("Project Directory: ", PROJECT_ROOT)

def preprocess_text(text:str):
    text = text.lower()
    # keep letter and number remove all remain
    reg_pattern = '[^a-z0-9A-Z_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễếệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ\s]'
    output = re.sub(reg_pattern, '', text)
    output = output.strip()
    return output

def load_context(clean_data_paths, input_datatype = 'txt'):
    context = []
    if input_datatype == 'txt':
        data_paths = []
        cxx_data_paths = glob.glob(clean_data_paths)
        cxx_data_paths.sort()
        for cxx_data_path in cxx_data_paths:
            data_path = glob.glob(cxx_data_path + '/*.txt')
            data_path.sort(reverse=False, key=lambda s:int(s[-7:-4]))
            data_paths += data_path
        for path in data_paths:
            with open(path, 'r', encoding='utf-8') as f:
                data = f.readlines()
                data = [item.strip() for item in data]
                context += data
    elif input_datatype == 'json':
        context_paths = glob.glob(clean_data_paths)
        context_paths.sort()
        for cxx_context_path in context_paths:
            paths = glob.glob(cxx_context_path + '/*.json')
            paths.sort(reverse=False, key=lambda x: int(x[-8:-5]))
            for path in paths:
                with open(path) as f:
                    context += [preprocess_text(' '.join(line)) for line in json.load(f)]
    else:
        print(f'not support reading the {input_datatype}')
        sys.exit()
    return context

def TfIdfTransform(data_path, save_tfids_object_path,  update=False , all_datatype=None): 
    tfidf_transform = {}
    context_matrix = {}
    ngram_range = (1, 1)
    for datatype in all_datatype:
        print(f'processing {datatype}')
        data_type_path = os.path.join(PROJECT_ROOT, data_path[datatype])
        print(f'load {datatype} context data from {data_type_path}')
        context = load_context(data_type_path)
        if update:
            print(f'load {datatype} tfidf object and matrix')
            tfidf_transform_path = os.path.join(PROJECT_ROOT, save_tfids_object_path, f'tfidf_transform_{datatype}.pkl')
            context_matrix_path = os.path.join(PROJECT_ROOT, save_tfids_object_path, f'sparse_context_matrix_{datatype}.npz')

            with open(tfidf_transform_path, 'rb') as f:
                old_tfidf_transformer = pickle.load(f)
            old_tfidf_matrix = load_npz(context_matrix_path)

            print(f'update {datatype} tfidf object and matrix')
            new_tfidf_matrix = old_tfidf_transformer.transform(context)
            context_matrix[datatype] = scipy.sparse.vstack([old_tfidf_matrix, new_tfidf_matrix])

        else:
            print(f'create {datatype} tfidf object and matrix')
            tfidf_transform[datatype] = TfidfVectorizer(input = 'content', ngram_range = ngram_range, token_pattern=r"(?u)\b[\w\d]+\b")
            context_matrix[datatype] = tfidf_transform[datatype].fit_transform(context)

        tfidf_transform_path = os.path.join(PROJECT_ROOT, save_tfids_object_path, f'tfidf_transform_{datatype}_test.pkl')
        context_matrix_path = os.path.join(PROJECT_ROOT, save_tfids_object_path, f'sparse_context_matrix_{datatype}_test.npz')

        os.makedirs(os.path.dirname(tfidf_transform_path), exist_ok=True)
        os.makedirs(os.path.dirname(context_matrix_path), exist_ok=True)
        print(f'save tfidf object to : {tfidf_transform_path}')
        with open(tfidf_transform_path, 'wb') as f:
            pickle.dump(tfidf_transform[datatype], f)

        save_npz(context_matrix_path, context_matrix[datatype])


def merge_tfidf_models(old_tfidf_pkl, old_tfidf_npz, new_data_paths, save_tfidf_pkl, save_tfidf_npz):
    ngram_range = (1, 1)
    # Bước 1: Tải mô hình TF-IDF cũ và ma trận TF-IDF cũ
    with open(old_tfidf_pkl, 'rb') as f:
        old_tfidf_transformer = pickle.load(f)
    old_tfidf_matrix = sp.load_npz(old_tfidf_npz)

    # Bước 2: Tạo mô hình TF-IDF mới với dữ liệu mới
    new_context = load_context(new_data_paths)
    new_tfidf_transformer = TfidfVectorizer(ngram_range=ngram_range, token_pattern=r"(?u)\b[\w\d]+\b")
    new_tfidf_matrix = new_tfidf_transformer.fit_transform(new_context).tocsr()

    # Bước 3: Kết hợp từ điển từ vựng của hai mô hình
    old_vocab = old_tfidf_transformer.vocabulary_
    new_vocab = new_tfidf_transformer.vocabulary_

    # Tạo từ điển gộp
    merged_vocab = {**old_vocab, **{word: i + len(old_vocab) for word, i in new_vocab.items() if word not in old_vocab}}

    # Tạo ma trận TF-IDF cho từ điển gộp
    combined_old_tfidf_matrix = sp.hstack([
        old_tfidf_matrix[:, old_vocab[word]] if word in old_vocab else sp.csr_matrix((old_tfidf_matrix.shape[0], 1))
        for word in merged_vocab.keys()
    ])
    
    combined_new_tfidf_matrix = sp.hstack([
        new_tfidf_matrix[:, new_vocab[word] + len(old_vocab)] if word in new_vocab else sp.csr_matrix((new_tfidf_matrix.shape[0], 1))
        for word in merged_vocab.keys()
    ])

    # Gộp hai ma trận TF-IDF
    combined_tfidf_matrix = sp.vstack([combined_old_tfidf_matrix, combined_new_tfidf_matrix])

    # Bước 4: Lưu mô hình TF-IDF gộp
    with open(save_tfidf_pkl, 'wb') as f:
        pickle.dump(TfidfVectorizer(vocabulary=merged_vocab), f)

    sp.save_npz(save_tfidf_npz, combined_tfidf_matrix)


def main():
    data_path = {
                # 'bbox':'dict/context_encoded/bboxes_encoded/*',
                # 'class':'dict/context_encoded/classes_encoded/*',
                # 'color':'dict/context_encoded/colors_encoded/*',
                # 'tag':'dict/context_encoded/tags_encoded/*',
                # 'number':'dict/context_encoded/number_encoded/*',
                "caption": "D:/Wordspace/Python/paper_competition/AI_Challenge/media/context_encoded_extra/caption_encoded_extra/*",
            } 
    save_tfids_object_path = "D:/Wordspace/Python/paper_competition/AI_Challenge/media/contexts_bin/"
    update = False
    all_datatype = ['caption']
    TfIdfTransform(data_path, save_tfids_object_path, update, all_datatype)

if __name__ == '__main__':
    main()
