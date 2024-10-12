# !python -m spacy download en_core_web_sm
import spacy
from openai import OpenAI
import os

class ExtractObjectsQuery:
    def __init__(self, path="en_core_web_sm",api="OPENAI_API_KEY"):
        self.nlp = spacy.load(path)
        self.client = OpenAI(api_key=os.getenv(api))

    def extract_objects(self, text):
        doc = self.nlp(text)
        main_objects = []

        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']: 
                main_objects.append(token.text)

        return main_objects
    
    def extract_objects_LLM(self, text):
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are my assistant to help me extract key objects in text sentences. Just return me a list in python , example: ['cat', 'dog', 'tree']"},
                {"role": "user", "content": text}
            ]
        )
        return completion.choices[0].message.content