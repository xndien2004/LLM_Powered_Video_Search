import json
import torch
import clip
import faiss
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.nlp_processing import translate

class MyFaissTextSearch:
    def __init__(self, bin_clip_file: str, json_path: str):    
        self.index_clip = self.load_bin_file(bin_clip_file)
        self.id2img_fps = self.load_json_file(json_path)
        self.translater = translate()
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/32", device=self.__device)

    def load_json_file(self, json_path: str):
        with open(json_path, 'r') as f: 
            js = json.load(f)
        return {int(k):v for k,v in js.items()}
    
    def load_bin_file(self, bin_file: str):
        return faiss.read_index(bin_file)

    def text_search(self, text, index, k=5):
        text = self.translater(text)
        print("Văn bản sau khi dịch : ", text)

        ###### TEXT FEATURES EXTRACTING ######
        text = clip.tokenize([text]).to(self.__device)  
        text_features = self.clip_model.encode_text(text)
        
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().detach().numpy().astype(np.float32)

        ###### SEARCHING #####
        index_choosed = self.index_clip
        
        if index is None:
            scores, idx_image = index_choosed.search(text_features, k=k)
        else:
            id_selector = faiss.IDSelectorArray(index)
            scores, idx_image = index_choosed.search(text_features, k=k, 
                                                     params=faiss.SearchParametersIVF(sel=id_selector))
        idx_image = idx_image.flatten()
        print("Ket qua tim kiem : ", idx_image)

        ###### GET INFOS KEYFRAMES_ID ######
        infos_query = list(map(self.id2img_fps.get, list(idx_image)))
        image_paths = [info['image_path'] for info in infos_query]
        return scores.flatten(), idx_image, infos_query, image_paths
        
def main():

    ##### TESTING #####
    bin_file='Data/faiss_clip.bin'
    json_path = 'Data/id2img_fps.json'

    cosine_faiss = MyFaissTextSearch(bin_file, json_path)

    ##### TEXT SEARCH #####
    text = 'Người nghệ nhân đang tô màu cho chiếc mặt nạ một cách tỉ mỉ. \
            Xung quanh ông là rất nhiều những chiếc mặt nạ. \
            Người nghệ nhân đi đôi dép tổ ong rất giản dị. \
            Sau đó là hình ảnh quay cận những chiếc mặt nạ. \
            Loại mặt nạ này được gọi là mặt nạ giấy bồi Trung thu.'
    text = 'Hình ảnh là cảnh trong bản tin HTV7 HD. Người dẫn chương trình mặc áo xanh và cà vạt chấm bi, đứng trước nền thành phố lúc hoàng hôn. Dưới cùng có dải chữ chạy thông tin sự kiện.'
    scores, _, infos_query, image_paths = cosine_faiss.text_search(text, None, k=10)
    print(scores, infos_query, image_paths)

if __name__ == "__main__":
    main()
