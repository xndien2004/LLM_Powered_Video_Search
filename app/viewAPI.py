from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
from utils import faiss_search,langchain_search, combine_search
from utils.LLM import llm_retrieval, llm
from .data_utils import *
from AIC.settings import MEDIA_ROOT, STATICFILES_DIRS
import os 
import time
import json
import gc

# setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# features preparing
tag_corpus_path = MEDIA_ROOT + '/tag/tag_corpus.txt'
dict_path = {
    'faiss_openclip_bin_path': MEDIA_ROOT + '/faiss_openclip.bin',
    'faiss_evalip_bin_path': MEDIA_ROOT + '/faiss_EVACLIP.bin',
    'id2img_fps_json_path': MEDIA_ROOT + '/id2img_fps.json',
    'map_asr_json_path': MEDIA_ROOT + '/map-asr.json',
    'dict_pkl_object_path': {
        'number': '/pkl/tfidf_transform_number.pkl',
        'number_tag': '/pkl/tfidf_transform_number_tag.pkl',
        'synthetic': '/pkl/tfidf_transform_synthetic.pkl'
    },
    'dict_npz_object_path': {
        'number': '/npz/sparse_context_matrix_number.npz',
        'number_tag': '/npz/sparse_context_matrix_number_tag.npz',
        'synthetic': '/npz/sparse_context_matrix_synthetic.npz'
    },
    'pkl_caption_path': '/pkl/tfidf_transform_caption.pkl',
    'npz_caption_path': '/npz/sparse_context_matrix_caption.npz',
    'dict_pkl_ocr_path': '/pkl/tfidf_transform_ocr.pkl',
    'dict_npz_ocr_path': '/npz/sparse_context_matrix_ocr.npz',
    'dict_pkl_asr_path': '/pkl/tfidf_transform_asr.pkl',
    'dict_npz_asr_path': '/npz/sparse_context_matrix_asr.npz',
    'dict_pkl_tag_path': '/pkl/tfidf_transform_tag.pkl',
    'dict_npz_tag_path': '/npz/sparse_context_matrix_tag.npz'
}

dict_path_extra = {
    'faiss_openclip_bin_path': MEDIA_ROOT + '/faiss_openclip_extra.bin',
    'faiss_evalip_bin_path': MEDIA_ROOT + '/faiss_EVAclip_extra.bin',
    'id2img_fps_json_path': MEDIA_ROOT + '/id2img_fps_extra.json',
    'map_asr_json_path': MEDIA_ROOT + '/map-asr.json',
    'dict_pkl_object_path': {
        'number': '/pkl_extra/tfidf_transform_number.pkl',
        'number_tag': '/pkl_extra/tfidf_transform_number_tag.pkl',
        'synthetic': '/pkl_extra/tfidf_transform_synthetic.pkl'
    },
    'dict_npz_object_path': {
        'number': '/npz_extra/sparse_context_matrix_number.npz',
        'number_tag': '/npz_extra/sparse_context_matrix_number_tag.npz',
        'synthetic': '/npz_extra/sparse_context_matrix_synthetic.npz'
    },
    'pkl_caption_path': '/pkl_extra/tfidf_transform_caption.pkl',
    'npz_caption_path': '/npz_extra/sparse_context_matrix_caption.npz',
    'dict_pkl_ocr_path': '/pkl_extra/tfidf_transform_ocr.pkl',
    'dict_npz_ocr_path': '/npz_extra/sparse_context_matrix_ocr.npz',
    'dict_pkl_asr_path': '/pkl/tfidf_transform_asr.pkl',
    'dict_npz_asr_path': '/npz/sparse_context_matrix_asr.npz',
    'dict_pkl_tag_path': '/pkl_extra/tfidf_transform_tag.pkl',
    'dict_npz_tag_path': '/npz_extra/sparse_context_matrix_tag.npz'
}

# load file
is_extra = "no"
is_openclip = True
is_evalip = False
is_object = False
if is_extra == "no":
    cosine_faiss = faiss_search.FaissSearch(dict_path, is_openclip, is_object, is_evalip)
elif is_extra == "yes":
    cosine_faiss_extra = faiss_search.FaissSearch(dict_path_extra, is_openclip, is_object, is_evalip)
else:
    cosine_faiss = faiss_search.FaissSearch(dict_path, is_openclip, is_object, is_evalip)
    cosine_faiss_extra = faiss_search.FaissSearch(dict_path_extra, is_openclip, is_object, is_evalip)

# LLM
key_api = "D:/Wordspace/Python/paper_competition/key_gpt.txt"
keyword = "./keyword.txt"
llm_auto = llm_retrieval.QueryProcessor(api_key_path=key_api)

# API 
class Objects(APIView):
    def get(self, request): 
        objects_path = STATICFILES_DIRS[0] + "/image/objects" 
        # kiểm tra thư mục objects_path có tồn tại không
        if not os.path.exists(objects_path):
            return Response({"error":"objects not found"}, status=status.HTTP_404_NOT_FOUND)
        objects = os.listdir(objects_path)

        image_objects = [
            {
                'src': f'/static/image/objects/{name}',
                'name': name.replace('.png', '').replace('_', ' ').title()  
            }
            for name in objects
        ]
        return Response({"image_objects":image_objects}, status=status.HTTP_200_OK)
class Colors(APIView):
    def get(self, request): 
        objects_path = STATICFILES_DIRS[0] + "/image/color" 
        # kiểm tra thư mục objects_path có tồn tại không
        if not os.path.exists(objects_path):
            return Response({"error":"objects not found"}, status=status.HTTP_404_NOT_FOUND)
        objects = os.listdir(objects_path)

        image_objects = [
            {
                'src': f'/static/image/color/{name}',
                'name': name.replace('.png', '').replace('_', '').title()  
            }
            for name in objects
        ]
        return Response({"image_objects":image_objects}, status=status.HTTP_200_OK)


# Query
class TextSearchImage(APIView): 
    def post(self, request):
        text = request.data.get('text')
        number = int(request.data.get('number'))
        choice_dict = request.data.get('choice_dict')
        re_rank = int(request.data.get('reRankNextScene_value')) 
        source_query = request.data.get('source_query')
        choice_extra = request.data.get('choiceExtraDict')
        options = request.data.get('options')
        print("options: ", options)
        idx_image = request.data.get("idx_IMG")
        idx_image = [int(id) for id in idx_image]

        start = time.time()
        list_results = []

        
        extra_index, noextra_index = get_extra_noextra(idx_image, source_query)
        extra_index = extra_index if re_rank else None
        noextra_index = noextra_index if re_rank else None
        search_k = len(idx_image) if re_rank else number

        search_methods = [
            ("ocr", "ocr_search"),
            ("openclip", "text_search_openclip"),
            ("evalip", "text_search_evalip"),
            ("caption", "caption_search")
        ]

        def run_all_searches(source, search_func_module, index):
            for choice_key, search_func_name in search_methods:
                search_func = getattr(search_func_module, search_func_name)
                run_search(
                    choice_key=choice_key, 
                    search_func=search_func, 
                    source=source, 
                    choice_dict=choice_dict, 
                    text=text, 
                    index=index, 
                    k=search_k, 
                    list_results=list_results
                )
        
        def run_search_asr(source, options, choice_dict, text, index_asr, search_k, list_results, search_func):
            if choice_dict.get("asr"):
                index_asr = get_index_image(cosine_faiss.map_asr, options)
                scores, idx_image, frame_idxs, image_paths = search_func.asr_search(text, k=search_k, index=index_asr)
                list_results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), [source]*len(frame_idxs)))

        def run_search_asr_not_index(source, choice_dict, text, search_k, list_results, search_func):
            if choice_dict.get("asr"):
                scores, idx_image, frame_idxs, image_paths = search_func.asr_search(text, k=search_k, index=None)
                list_results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), [source]*len(frame_idxs)))
        if options == "All":
            if choice_extra.get("extra"):
                run_all_searches("extra", cosine_faiss_extra, extra_index)
                run_search_asr_not_index("extra", choice_dict, text, search_k, list_results, cosine_faiss_extra)
            if choice_extra.get("no_extra"):
                run_all_searches("no_extra", cosine_faiss, noextra_index)
                run_search_asr_not_index("no_extra", choice_dict, text, search_k, list_results, cosine_faiss)
        else:
             
            if choice_extra.get("extra"):
                index = get_index_image(cosine_faiss_extra.id2img_fps, options)
                run_all_searches("extra", cosine_faiss_extra, index)
                run_search_asr("extra", options, choice_dict, text, index, search_k, list_results, cosine_faiss_extra)
            if choice_extra.get("no_extra"):
                index = get_index_image(cosine_faiss.id2img_fps, options)
                run_all_searches("no_extra", cosine_faiss, index)
                run_search_asr("no_extra", options, choice_dict, text, index, search_k, list_results, cosine_faiss)


        # Combine the search results
        scores, idx_image, frame_idxs, image_paths, source = combine_search.combined_ranking_score(list_results, topk=search_k, alpha=0.7, beta=0.3)
        print("list_results: ", len(list_results))

        # Propose videos
        video_propose = sort_by_frequent_video(scores, idx_image, frame_idxs, image_paths)
        print("video_propose: ", len(video_propose))

        end = time.time()
        print("Time search: ", end - start)
        print("total quantity: ",len(idx_image))
 
        # propose video
        video_propose = sort_by_frequent_video(scores, idx_image, frame_idxs, image_paths)
        print("video_propose: ", len(video_propose))
        return Response({"scores":scores, "idx_image":idx_image, "frame_idxs":frame_idxs, 
                         "image_paths":image_paths, "video_propose":video_propose, "source":source}, status=status.HTTP_200_OK)

        
class FilterSearch(APIView):
    def post(self, request):
        filter_dict = request.data.get('filter_dict')["query"]
        number = int(request.data.get('numberSearch'))
        next_objects = request.data.get("next_objects")
        source_query = request.data.get('source_query')
        choice_extra = request.data.get('choiceExtraDict')
        id_img_objects = request.data.get("id_img_objects")
        id_img_objects = [int(id) for id in id_img_objects]

        extra_index, noextra_index = get_extra_noextra(id_img_objects, source_query)
        extra_index = extra_index if next_objects else None
        noextra_index = noextra_index if next_objects else None
        search_k = len(id_img_objects) if next_objects else number
        
        list_results = []
        start = time.time()
        if choice_extra.get("extra"):
            scores, idx_image, frame_idxs, image_paths = cosine_faiss_extra.context_search(filter_dict, k=search_k, index=extra_index)
            list_results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), ["extra"]*len(frame_idxs)))
        if choice_extra.get("no_extra"):
            scores, idx_image, frame_idxs, image_paths = cosine_faiss.context_search(filter_dict, k=search_k, index=noextra_index)
            list_results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), ["no_extra"]*len(frame_idxs)))

        # Combine the search results
        scores, idx_image, frame_idxs, image_paths, source = combine_search.combined_ranking_score(list_results, topk=search_k, alpha=0.7, beta=0.3)
        video_propose = sort_by_frequent_video(scores, idx_image, frame_idxs, image_paths)
        return Response({"scores":scores, "idx_image":idx_image, "frame_idxs":frame_idxs, 
                         "image_paths":image_paths, "video_propose":video_propose, "source":source}, status=status.HTTP_200_OK)

    
class ImageQuery(APIView):
    def post(self, request):
        start = time.time()
        image_file = request.FILES.get('image') 
        number = int(request.data.get('numberSearch'))
        next_objects = int(request.data.get("nextScene"))
        source_query = request.data.get('source_query')
        choice_extra = json.loads(request.data.get('choiceExtraDict'))
        id_Img = request.data.get("id_Img") 
        id_Img = id_Img.split(",")
        id_Img = [int(id) for id in id_Img]

        image_path = save_image(image_file, format='.jpg')
        
        extra_index, noextra_index = get_extra_noextra(id_Img, source_query)
        extra_index = extra_index if next_objects else None
        noextra_index = noextra_index if next_objects else None
        search_k = len(id_Img) if next_objects else number

        list_results = []
        if choice_extra.get("extra"):
            scores, idx_image, frame_idxs, image_paths = cosine_faiss_extra.image_search(image_path, k=search_k, index=extra_index)
            list_results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), ["extra"]*len(frame_idxs)))
        if choice_extra.get("no_extra"):
            scores, idx_image, frame_idxs, image_paths = cosine_faiss.image_search(image_path, k=search_k, index=noextra_index)
            list_results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), ["no_extra"]*len(frame_idxs)))

        # Combine the search results
        scores, idx_image, frame_idxs, image_paths, source = combine_search.combined_ranking_score(list_results, topk=search_k, alpha=0.7, beta=0.3)
        end = time.time()
        print("Time image search : ", end - start)
        os.remove(image_path) 

        video_propose = sort_by_frequent_video(scores, idx_image, frame_idxs, image_paths)
        
        return Response({"scores":scores, "idx_image":idx_image, "frame_idxs":frame_idxs, 
                         "image_paths":image_paths, "video_propose":video_propose, "source":source}, status=status.HTTP_200_OK)
    
class TagQuery(APIView):
    def get(self, request):
        tag_corpus = cosine_faiss.load_txt_file(tag_corpus_path)

        tags = [tag.strip() for tag in tag_corpus]
        return Response({"tags":tags}, status=status.HTTP_200_OK)
    
    def post(self, request):
        start = time.time()
        tag_query = request.data.get('tag_query') 
        number = int(request.data.get('numberSearch'))
        next_scene = request.data.get("nextScene")
        source_query = request.data.get('source_query')
        choice_extra = request.data.get('choiceExtraDict')
        id_img = request.data.get('id_Img')
        id_img = [int(id) for id in id_img]

        extra_index, noextra_index = get_extra_noextra(id_img, source_query)
        extra_index = extra_index if next_scene else None
        noextra_index = noextra_index if next_scene else None
        search_k = len(id_img) if next_scene else number

        list_results = []
        if choice_extra.get("extra"):
            scores, idx_image, frame_idxs, image_paths = cosine_faiss_extra.tag_search(tag_query, k=search_k, index=extra_index)
            list_results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), ["extra"]*len(frame_idxs)))
        if choice_extra.get("no_extra"):
            scores, idx_image, frame_idxs, image_paths = cosine_faiss.tag_search(tag_query, k=search_k, index=noextra_index)
            list_results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), ["no_extra"]*len(frame_idxs)))

        # Combine the search results
        scores, idx_image, frame_idxs, image_paths, source = combine_search.combined_ranking_score(list_results, topk=search_k, alpha=0.7, beta=0.3)
        end = time.time()
        print("Time tag search : ", end - start)
        video_propose = sort_by_frequent_video(scores, idx_image, frame_idxs, image_paths)
        return Response({"scores":scores, "idx_image":idx_image, "frame_idxs":frame_idxs, 
                         "image_paths":image_paths, "video_propose":video_propose, "source":source}, status=status.HTTP_200_OK)
    
# cluster
class ImageSearchCluster(APIView):
    def post(self, request):
        start = time.time()
        name_frame = request.data.get('name_frame')
        id_video = request.data.get('id_video')
        choice_extra = request.data.get('choiceExtraDict')

        LO_folder = id_video.split('_')[0]
        image_path = MEDIA_ROOT +f'/Keyframes/Keyframes_{LO_folder}/keyframes/' + id_video + '/' + name_frame

        list_results = []
        if choice_extra.get("extra"):
            scores, idx_image, frame_idx, image_paths = cosine_faiss_extra.image_search(image_path, k=100)
            list_results.append((scores, idx_image, frame_idx, image_paths, list(range(1, len(frame_idx) + 1)), ["extra"]*len(frame_idx)))
        if choice_extra.get("no_extra"):
            scores, idx_image, frame_idx, image_paths = cosine_faiss.image_search(image_path, k=100)
            list_results.append((scores, idx_image, frame_idx, image_paths, list(range(1, len(frame_idx) + 1)), ["no_extra"]*len(frame_idx)))
        end = time.time() 

        # Combine the search results
        scores, idx_image, frame_idx, image_paths, source = combine_search.combined_ranking_score(list_results, topk=100, alpha=0.7, beta=0.3)
        print("Time image search : ", end - start) 
        return Response({"idx_frame":frame_idx,"image_paths":image_paths, "source":source}, status=status.HTTP_200_OK)
class ClusterFrames(APIView): 
    def post(self, request):
        videoId = request.data.get('videoId')
        index_frame = request.data.get('index_frame')

        LO_folder = videoId.split('_')[0]
        video_path = str(MEDIA_ROOT + f'/Keyframes/Keyframes_{LO_folder}/keyframes/' + videoId)
        if not os.path.exists(video_path):
            return Response({"error":"videoId not found"}, status=status.HTTP_404_NOT_FOUND)
        frames = os.listdir(video_path)
 
        cluster_frames = get_n_cluster_frames(30, index_frame, frames)
        ID_frame = cluster_frames.index(index_frame)
        # get path of cluster frames
        cluster_frames_path = [f'/Keyframes/Keyframes_{LO_folder}/keyframes/'+ videoId + '/' + frame for frame in cluster_frames]

        video_info, start_time = get_video_info(videoId, index_frame.split('.')[0])
        if video_info is None:
            return Response({"error":"video info not found"}, status=status.HTTP_404_NOT_FOUND)

        print("start_time: ", start_time)
        return Response({"cluster_frames_path":cluster_frames_path, "video_info":video_info, "start_time":start_time,
                         "ID_frame":ID_frame}, status=status.HTTP_200_OK)
    
# feelback
class Feelback(APIView):
    def post(self, request):
        idx_image = request.data.get('idx_image')
        idx_image = [int(idx) for idx in idx_image]
        scores = request.data.get('scores')
        scores = [float(score) for score in scores]
        positive_feedback_idxs = request.data.get('live_active')
        negative_feedback_idxs = request.data.get('dislive_active')
        source_query = request.data.get('source_query')
        choice_extra = request.data.get('choiceExtraDict')
        
        list_results = []
        extra_index, noextra_index = get_extra_noextra(idx_image, source_query)
        extra_scores, noextra_scores = get_extra_noextra(scores, source_query)
        if choice_extra.get("extra"):
            previous_results = [{"video_info":{"lst_idxs":extra_index, "lst_scores":extra_scores}}]
            positive_feedback_idxs = list(set(positive_feedback_idxs) & set(extra_index))
            negative_feedback_idxs = list(set(negative_feedback_idxs) & set(extra_index))
            scores, idx_image, frame_idxs, image_paths = cosine_faiss_extra.feelback(previous_results, positive_feedback_idxs, negative_feedback_idxs)
            list_results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), ["extra"]*len(frame_idxs)))
        if choice_extra.get("no_extra"):
            previous_results = [{"video_info":{"lst_idxs":noextra_index, "lst_scores":noextra_scores}}]
            positive_feedback_idxs = list(set(positive_feedback_idxs) & set(noextra_index))
            negative_feedback_idxs = list(set(negative_feedback_idxs) & set(noextra_index))
            scores, idx_image, frame_idxs, image_paths = cosine_faiss.feelback(previous_results, positive_feedback_idxs, negative_feedback_idxs)
            list_results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), ["no_extra"]*len(frame_idxs)))

        # Combine the search results
        scores, idx_image, frame_idxs, image_paths, source = combine_search.combined_ranking_score(list_results, topk=None, alpha=0.7, beta=0.3)
        video_propose = sort_by_frequent_video(scores, idx_image, frame_idxs, image_paths)
        return Response({"scores":scores, "idx_image":idx_image, "frame_idxs":frame_idxs, 
                         "image_paths":image_paths, "video_propose":video_propose, "source":source}, status=status.HTTP_200_OK)


# LLM 
class LLM(APIView):
    def post(self, request):
        query = request.data.get('query')
        number = int(request.data.get('number'))

        datas = llm_auto.get_query_variants(query)
        list_results = []
        for data in datas:
            method = data.get('method')
            query = data.get('query')
            results = []
            if 'ocr' in method:
                scores, idx_image, frame_idxs, image_paths = cosine_faiss.ocr_search(query, k=number, index=None)
                results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), ["no_extra"]*len(frame_idxs)))
            if 'asr' in method:
                scores, idx_image, frame_idxs, image_paths = cosine_faiss.asr_search(query, k=number, index=None)
                results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), ["no_extra"]*len(frame_idxs)))
            if 'openclip' in method:
                scores, idx_image, frame_idxs, image_paths = cosine_faiss.text_search_openclip(query, k=number, index=None)
                results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), ["no_extra"]*len(frame_idxs)))
            if 'evalip' in method:
                scores, idx_image, frame_idxs, image_paths = cosine_faiss.text_search_evalip(query, k=number, index=None)
                results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), ["no_extra"]*len(frame_idxs)))
            if 'caption' in method:
                scores, idx_image, frame_idxs, image_paths = cosine_faiss.caption_search(query, k=number, index=None)
                results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), ["no_extra"]*len(frame_idxs)))
            scores, idx_image, frame_idxs, image_paths, source = combine_search.combined_ranking_score(results, topk=number, alpha=0.7, beta=0.3)
            list_results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), source))
        scores, idx_image, frame_idxs, image_paths, source = combine_search.combined_ranking_score(list_results, topk=number, alpha=0.7, beta=0.3)
        video_propose = sort_by_frequent_video(scores, idx_image, frame_idxs, image_paths)
        return Response({"scores":scores, "idx_image":idx_image, "frame_idxs":frame_idxs, 
                         "image_paths":image_paths, "video_propose":video_propose, "source":source}, status=status.HTTP_200_OK)
    
class LLMChatbot(APIView):
    def post(self, request):
        message = str(request.data.get('message'))
        
        model = "gpt-4o"
        temperature = 0.7
        history = []
        search = cosine_faiss
        text_path = keyword
        bot = llm.get_llm(key_api=key_api, model=model, temperature=temperature, 
                               history=history, search=search, text_path=text_path) 
        bot_response = bot.generate_answer(message)
        return Response({"reply":bot_response["reply"],
                         "frame_idxs":bot_response["frame_idxs"],
                         "image_paths":bot_response["image_paths"]}, status=status.HTTP_200_OK)