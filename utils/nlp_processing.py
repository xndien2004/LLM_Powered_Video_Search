import httpcore
setattr(httpcore, 'SyncHTTPTransport', 'AsyncHTTPProxy')
import google.generativeai as genai
import googletrans
from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()


def translate(text, to_lang='en'):
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


# Testing
text = "Xin chào, tôi là một sinh viên đại học"

try : 
    print(translate_OpenAI(text))
    print("Dịch bằng OpenAI")
except :
    try :
        print(translate_Gemini(text))
        print("Dịch bằng Gemini")
    except :
        print(translate(text))
        print("Dịch bằng thư viện Google Translate")