import httpcore
setattr(httpcore, 'SyncHTTPTransport', 'AsyncHTTPProxy')
import google.generativeai as genai
import googletrans
from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()


def translate_lib(text, to_lang='en'):
    """
    Translates the input text from the source language to the target language.

    :param text: The text to be translated
    :param to_lang: The language code of the target language (default is English 'en')
    :return: The translated text.
    """
    translator = googletrans.Translator()
    # text = text.lower()
    print("Translation by Google Translate Library")
    return translator.translate(text, dest=to_lang).text # auto detect source language and translate to target language

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def translate_OpenAI(text, to_lang = 'en'):
    """
    Translates the input text from the source language to the target language using OpenAI API.

    :param text: The text to be translated
    :param to_lang: The language code of the target language (default is English 'en')
    :return: The translated text.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", # gpt-3.5-turbo, gpt-4o, ...
        messages= [
            {
                "role": "system",
                "content": f"You will be given a piece of text and your task is to translate it into {to_lang}"
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=0.7,
        max_tokens=1000,
        top_p=1
    )
    print("Translation by OpenAI")
    return response.choices[0].message.content


GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

def translate_Gemini(text, to_lang = 'en'):
    """
    Translates the input text from the source language to the target language using OpenAI API.

    :param text: The text to be translated
    :param to_lang: The language code of the target language (default is English 'en')
    :return: The translated text.
    """
    model = genai.GenerativeModel('gemini-pro') #gemini-pro-vision , text-bison-001
    chat = model.start_chat(history=[
        {
            'role': 'user',
            'parts': [f"You will be given a piece of text and your task is to translate it into {to_lang}"]
        },
    ])
    response = chat.send_message(text)
    print("Translation by Gemini")
    return response.text


# def translate(text, to_lang='en'):
#     try:
#         translation = translate_OpenAI(text, to_lang)
#         print("Translated by OpenAI")
#     except Exception as e1:
#         print(f"OpenAI translation failed: {e1}")
#         try:
#             translation = translate_Gemini(text, to_lang)
#             print("Translated by Gemini")
#         except Exception as e2:
#             print(f"Gemini translation failed: {e2}")
#             translation = translate_lib(text, to_lang)
#             print("Translated by Google Translate Library")
#     return translation

# Testing
# text = "Xin chào, tôi là một sinh viên đại học"
# print(translate(text))

class translate:
    def __init__(self):
        # Initialize any resources required for translation here
        pass
    
    def __call__(self, text,to_lang='en'):
        to_lang = 'en'
        try:
            translation = translate_OpenAI(text, to_lang)
            print("Translated by OpenAI")
        except Exception as e1:
            print(f"OpenAI translation failed: {e1}")
            try:
                translation = translate_Gemini(text, to_lang)
                print("Translated by Gemini")
            except Exception as e2:
                print(f"Gemini translation failed: {e2}")
                translation = translate_lib(text, to_lang)
                print("Translated by Google Translate Library")
        return translation