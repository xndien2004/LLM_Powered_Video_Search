import google.generativeai as genai
from deep_translator import GoogleTranslator
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
    translation = GoogleTranslator(source='auto', target=to_lang).translate(text)
    print("Translation by Google Translate Library")
    return translation


def translate_OpenAI(text, to_lang = 'en'):
    """
    Translates the input text from the source language to the target language using OpenAI API.

    :param text: The text to be translated
    :param to_lang: The language code of the target language (default is English 'en')
    :return: The translated text.
    """
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
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



def translate_Gemini(text, to_lang = 'en'):
    """
    Translates the input text from the source language to the target language using OpenAI API.

    :param text: The text to be translated
    :param to_lang: The language code of the target language (default is English 'en')
    :return: The translated text.
    """
    GOOGLE_API_KEY=os.getenv('AIzaSyAoAU_SQ0Z9G7tNm-5GNoCYJqmsqpZOjgM')
    genai.configure(api_key=GOOGLE_API_KEY)
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
