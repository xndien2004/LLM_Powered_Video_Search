import os
import sys
import copy
import ast
import base64
import requests
import io
from PIL import Image
from AIC.settings import MEDIA_ROOT
from utils import combine_search
from utils.nlp_processing import translate_lib

from openai import OpenAI

# MEDIA_ROOT = r"D:\Wordspace\Python\paper_competition\AI_Challenge\media"
dataset_path = os.path.join(MEDIA_ROOT, 'dataset')
search_path = os.path.join(dataset_path, 'search')

def get_llm(key_api, model, temperature, history, search, text_path):
    if model == 'gpt-3.5-turbo':
        return OpenAiLlm(key_api=key_api, model=model, temperature=temperature, history=history, search=search, text_path=text_path,
                         image=True)
    elif model == 'gpt-4o':
        return OpenAiLlm(key_api=key_api, model=model, temperature=temperature, history=history, search=search, text_path=text_path,
                         image=True)
    elif model == 'dall-e-3':
        return DALLE(key_api=key_api,model=model, temperature=temperature)

class OpenAiLlm:
    def __init__(self, key_api, model, temperature, history, search, text_path, image):
        super(OpenAiLlm, self).__init__()
        self.history = history
        self.model = model
        self.temperature = temperature
        with open(key_api, 'r') as file:
            key_api = file.read().replace('\n', '')
        self.client = OpenAI(api_key=key_api)
        self.search = search
        self.text_path = text_path
        self.support_image = image

    def is_search(self, content):
        # messages = copy.deepcopy(self.history)
        messages = []
        messages.extend([
            {'role': 'user', 'content': content},
            {'role': 'system',
            'content': "Bạn là một trợ lý tìm kiếm hình ảnh dựa trên truy vấn của người dùng. Nhiệm vụ của bạn là xác định xem yêu cầu của người dùng có liên quan đến việc tìm kiếm hình ảnh hay không. "
                "Chỉ trả lời bằng 'có' hoặc 'không' dựa trên ý định của người dùng là tìm kiếm hình ảnh."
                "\nKhông đưa ra bất kỳ giải thích nào khác ngoại trừ 'có' hoặc 'không', chỉ trả lời 'có' hoặc 'không'."}
        ])
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        response = completion.choices[0].message.content.strip().lower()
        print("[LLM] is search: " + response)
        return response == 'có'

    def extract_keywords(self, content):
        messages = copy.deepcopy(self.history)
        messages.append({'role': 'user', 'content': content})    
        messages.extend([
                        {'role': 'user', 'content': content},
                        {'role': 'system',
                        'content': 'As a text analysis expert, please extract keywords from the historical dialogue '
                                    'that describe the types of images the user wants to find at this time. Keywords '
                                    'should be limited to adjectives or nouns, avoiding verbs and adverbs. '
                                    'Additionally, convert any negative keywords into their positive counterparts '
                                    'using synonyms. Output only the relevant keywords, separated by spaces without '
                                    'any punctuation. If no relevant keywords are found, output "none" or an empty '
                                    'string. For example, if the sentence is "I want red apples but not fresh," the '
                                    'output should be "red apples ripe." If there are no relevant keywords, output '
                                    'an empty string.'}
                    ])
        print("[LLM] extract keyword: " + str(messages))
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        keyword = completion.choices[0].message.content
        with open(self.text_path, 'w', encoding='utf-8') as file:
            file.write(keyword)
        print("[LLM] keywords: " + keyword)
        return keyword

    def reply(self, content, prompt, images=[]):
        messages = copy.deepcopy(self.history)
        self.history.append({'role': 'user', 'content': copy.deepcopy(content)})
        if self.support_image and len(images) != 0:
            content.extend(images)
        messages.append({'role': 'user', 'content': content})
        messages.append({'role': 'system', 'content': prompt})
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            max_tokens=500
        )
        response = completion.choices[0].message.content
        self.history.append({'role': 'assistant', 'content': response})
        while len(self.history) > 10:
            self.history.pop(0)
        return response

    def generate_answer(self, content):
        if self.is_search(content) == False:
            print("[LLM] is not search")
            reply = self.reply(
                content=content,
                images=[],
                prompt='You should engage in conversation with the user about their preferences and needs. ' +
                        'Use only plain text to describe the interaction. Output text only.')
            return {
                'frame_idxs': [],
                'image_paths': [],
                'reply': str(translate_lib(reply, to_lang='vi'))
            }
        keyword_content = [{"type": "text", "text": content}]
            
        keyword = self.extract_keywords(content=keyword_content)
        
        _, idx_image, frame_idxs, image_paths, _ = self.search_image(str(translate_lib(keyword, to_lang='vi')), 3)

        base64_images = [{}]*len(image_paths)
        content = [{"type": "text", "text": content}]
        for idx, img in enumerate(image_paths):
            image = Image.open(MEDIA_ROOT+img)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG") 
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            base64_images[idx] = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                },
            }
        content.extend(base64_images)

        reply = self.reply(
            content=content,
            images=base64_images,
            prompt="As a retrieval-augmented generation assistant, your task is to describe each image provided by "
                   "the system. When users provide a query, a retriever will source relevant images, and it is your "
                   "job to complement these images with appropriate text descriptions. The text you provide should "
                   "not introduce new attributes beyond the user's query but should enrich and provide context to "
                   "the images returned by the retriever. Your mission is to ensure that your descriptions align "
                   "seamlessly with the user's query and the vibe of the images. All responses should appear as one "
                   "brief and clear explanation to the user, enhancing their understanding and improving their overall "
                   "experience. Your sole responsibility is to offer relevant text corresponding to the user's query "
                   "for each image. Please refrain from mentioning any limitations regarding the provision of images "
                   "or your abilities. Focus solely on providing detailed and contextually appropriate descriptions "
                   "for each image in the order they are presented.")
        return {
            'frame_idxs': frame_idxs,
            'image_paths': image_paths,
            'reply': str(translate_lib(reply, to_lang='vi'))
        }

    def type_search(self, content):
        temp_prompt = '''
        Based on the content provided, determine the appropriate search method(s) from the following options: 
        ["OCR", "ASR", "openclip", "Caption"].

        - "OCR": Use if the query involves searching for visible text within images.
        - "ASR": Use if the query involves searching audio or video content for spoken words.
        - "openclip": Use if the query focuses on visual elements or characteristics within an image.
        - "Caption": Use if the query relates to metadata or descriptive text associated with the image.

        Return the result as a Python list of strings with no additional text. For example:
        
        Input: "Find images of a cat."
        Output: ["openclip", "Caption"]

        Input: "Search for documents with visible text."
        Output: ["OCR"]

        Ensure that the response is a valid Python list with only the method names in double quotes.
        '''
        
        reply = self.reply(
            content=content,
            prompt=temp_prompt)
        
        reply = ast.literal_eval(reply)  # Assuming reply will now be in a proper list format
        return reply

    
    def search_image(self, content, number):
        # list_search = self.type_search(content)
        list_search = "openclip"#[x.lower() for x in list_search]
        results = []
        if "ocr" in list_search:
            scores, idx_image, frame_idxs, image_paths = self.search.ocr_search(content, k=number, index=None)
            results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), ["no_extra"]*len(frame_idxs)))
        if 'asr' in list_search:
                scores, idx_image, frame_idxs, image_paths = self.search.asr_search(content, k=number, index=None)
                results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), ["no_extra"]*len(frame_idxs)))
        if 'openclip' in list_search:
                scores, idx_image, frame_idxs, image_paths = self.search.text_search_openclip(content, k=number, index=None)
                results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), ["no_extra"]*len(frame_idxs)))
        # if 'evalip' in list_search:
        #         scores, idx_image, frame_idxs, image_paths = self.search.text_search_evalip(content, k=number, index=None)
        #         results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), ["no_extra"]*len(frame_idxs)))
        if 'caption' in list_search:
                scores, idx_image, frame_idxs, image_paths = self.search.caption_search(content, k=number, index=None)
                results.append((scores, idx_image, frame_idxs, image_paths, list(range(1, len(frame_idxs) + 1)), ["no_extra"]*len(frame_idxs)))
        scores, idx_image, frame_idxs, image_paths, source = combine_search.combined_ranking_score(results, topk=number, alpha=0.7, beta=0.3)
        return scores, idx_image, frame_idxs, image_paths, source


class DALLE:
    def __init__(self, key_api, model, temperature):
        super(DALLE, self).__init__()
        self.model = model
        with open(key_api, 'r') as file:
            key_api = file.read().replace('\n', '')
        self.client = OpenAI(api_key=key_api)
        self.temperature = temperature

    def generate_answer(self, content):
        response = self.client.images.generate(
            model='dall-e-3',
            prompt=content,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        image_data = requests.get(image_url).content
        with open('D:/Wordspace/Python/paper_competition/AI_Challenge/media/generated_image.png', 'wb') as f:
            f.write(image_data)
        completion = self.client.chat.completions.create(
            model='gpt-4-turbo',
            temperature=self.temperature,
            max_tokens=500,
            messages=[
                {'role': 'user',
                 'content': [
                     {"type": "text", "text": content},
                     {
                         "type": "image_url",
                         "image_url": {
                             "url": image_url
                         },
                     },
                 ], },
                {'role': 'system',
                 'content': 'I want you to act as a retrieval-augmented generation assistant. ' +
                            'When users provide a query, there will a retriever sourcing relevant images, ' +
                            'and it is your job to complement them with appropriate text. ' +
                            'The text you provide shall not add a new attribute to the user\'s query, ' +
                            'but to enrich and provide context to the images returned by the retriever. ' +
                            'Your mission is to make sure that your text aligns seamlessly with the user\'s ' +
                            'query and vibe of the images. ' +
                            'All responses should appear as one brief and clear explanation to the user, ' +
                            'enhance their understanding and improve their overall experience. ' +
                            'Your sole responsibility is to offer relevant text corresponding to the user\'s query. ' +
                            'Please refrain from mentioning any limitations regarding the provision of images.'}
            ]
        )
        reply = completion.choices[0].message.content
        return {
            'frame_idxs': [""],
            'image_paths': ["/generated_image.png"],
            # 'images': [
            #     {
            #         'id': image_url,
            #     }
            # ],
            'reply': reply
        }
    
if __name__ == "__main__":
    key_api = r"D:\Wordspace\Python\paper_competition\key_gpt.txt"
    model = "gpt-3.5-turbo"
    temperature = 0.5
    history = []
    search = None
    text_path = r"D:\Wordspace\Python\paper_competition\AI_Challenge\AIC\app\keyword.txt"
    llm = get_llm(key_api, model, temperature, history, search, text_path)
    print(llm.is_search(""))