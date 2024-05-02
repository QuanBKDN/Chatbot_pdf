################Library#########################################################################################################################################
from pydub import AudioSegment
import fasttext as ft
import json
import fitz
import os
import numpy as np
import re
import threading
import azure.cognitiveservices.speech as speechsdk
import openai
from core.config import *
import pyaudio
import wave
import time
from pydub import AudioSegment
import soundfile as sf
from scipy.io import wavfile


################Function support for main##########################################################################################################################



def detect_language_prompt(text):
    model_file = 'lid.176.ftz'
    language_code = 'language_code.json'
    model = ft.load_model(model_file)
    id2lang = json.load(open(language_code,encoding='utf-8'))
    res = model.predict(text)
    language_code = res[0][0][9:]
    language = id2lang.get(language_code)

    return language


def convert_to_wav(input_file_path):
    # Load the input audio file
    audio_data = AudioSegment.from_file(input_file_path)    
    # Set the desired properties
    sample_width = 2
    frame_rate = 16000
    channels = 1
    converted_audio = audio_data.set_frame_rate(frame_rate)
    converted_audio = converted_audio.set_sample_width(sample_width)
    converted_audio = converted_audio.set_channels(channels)
    # Export the converted audio to WAV format
    converted_audio.export(input_file_path, format="wav")

def recognize_speech_from_wav(wav_file_path, azure_speech_key, azure_speech_region):
    languages = ["en-US", "ja-JP", "ko-KR"]
    speech_config = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_speech_region)
    audio_config = speechsdk.audio.AudioConfig(filename=wav_file_path)

    if languages:

        auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=languages)
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, 
            auto_detect_source_language_config=auto_detect_source_language_config,
            audio_config=audio_config)

    else:

        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config)
    recognized_text = ""

    def recognized_callback(evt):
        nonlocal recognized_text
        recognized_text += evt.result.text
        
    speech_recognizer.recognized.connect(recognized_callback)
    done_event = threading.Event()
    speech_recognizer.canceled.connect(lambda evt: done_event.set())
    speech_recognizer.session_stopped.connect(lambda evt: done_event.set())
    speech_recognizer.start_continuous_recognition()
    while not done_event.is_set():
        time.sleep(0.5)
    speech_recognizer.stop_continuous_recognition()

    return recognized_text

def pdf_to_text(file, start_page=1, end_page=None):
    text_list = []
    if (os.path.splitext(file)[1]).lower() == ".pdf":
        doc = fitz.open(file)
        total_pages = doc.page_count
        end_page = total_pages
        for i in range(start_page - 1, end_page):
            text = doc.load_page(i).get_text("text")
            text = preprocess(text)
            text_list.append(text)
        doc.close()

    return text_list


def get_chunks(path):
    pdf_files = [file for file in os.listdir(path) if ((os.path.splitext(file)[1]).lower() == ".pdf")]
    file_chunks = {}
    for pdf_file in pdf_files:
        pdf_path = os.path.join(path, pdf_file)
        text_list = pdf_to_text(pdf_path)
        chunks = text_to_chunks(text_list, pdf_path)
        key = os.path.splitext(pdf_file)[0]
        file_chunks[key]=chunks
        
    return file_chunks

def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    
    return text

def add_source_numbers(lst):
    refDoc = []
    for item in lst:
        number_page = int(item.split('] ')[1][1])
        name_doc = item.split('] ')[0][1:]
        new_item = '] '.join(item.split('] ')[2:])
        refDoc.append({"page": str(number_page),
                      "document_name": name_doc, "content": new_item})
        
    return refDoc

def text_to_chunks(texts, file, word_length=100, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    chunks = []
    name_file = file.split(os.path.sep)[-1]
    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i + word_length]
            if (i + word_length) > len(words) and (len(chunk) < word_length) and (len(text_toks) != (idx + 1)):
                text_toks[idx + 1] = chunk + text_toks[idx + 1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[{name_file}] [{idx + start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)

    return chunks

def detect_language_pdf(text):
    model_file = 'lid.176.ftz'
    language_code = 'language_code.json'
    model = ft.load_model(model_file)
    id2lang = json.load(open(language_code,encoding='utf-8'))
    res = model.predict(text)
    language_code = res[0][0][0][9:]

    return language_code

def save_pdf_to_folder(pdf_path, folder):
    filename = os.path.basename(pdf_path)
    new_pdf_path = os.path.join(folder, filename)
    os.rename(pdf_path, new_pdf_path)


def get_unique_filename(original_filename):
    base, extension = os.path.splitext(original_filename)
    count = 1

    while os.path.exists(original_filename):

        new_filename = f"{base}_{count}{extension}"
        count += 1

    return new_filename

def delete_file_npy(pathFile):
        try:
            pathFile_npy = pathFile.split("/")
            folder = "embedding_" + pathFile_npy[-2]
            file_npy = pathFile_npy[-1].replace(".pdf", ".npy")
            pathFile_ = os.path.join(
                pathFile[: pathFile.index(pathFile_npy[-2])], folder, file_npy)
            os.remove(pathFile_)
            message = "success"
            return message
        except:
            message = "failed"
            return message

########################################################################################################################################################################