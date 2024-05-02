import pyaudio
import openai
import tensorflow_hub as hub
import numpy as np
import tensorflow_text



####################################################################################
OPENAI_KEY = "24485445c9064badab3108635cc34f96" # DungPM3 key
TEMPERATURE = 0.6
MODELS = "chatgpt-test2"
MAX_TOKENS = 16000
DEBUG = True
PORT = 5000
HOST = '0.0.0.0'

####################################################################################
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

CHANNELS = 1
RATE = 44100
CHUNK = 1024
is_recording = False
openai.api_type = "azure"
openai.api_base = "https://ddx-chatgpt.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
azure_speech_key = "e5c0fa4315474df3ad4673f3d3f6e2c5"
azure_speech_region = "eastus"


####################################################################################

prompt_template = """
Search result:
----------------------
      {context_str}
----------------------
Instructions: Compose a comprehensive answer to the query using the search results provided. If the search question is not related to the provided knowledge (Search result) simply answer with the previously learned knowledge.
If the question related to the search results provided covers more than one topic with the same name, create a separate answer for each topic. Include only information found in search results and do not add any additional information. Make sure the answers are correct and do not output incorrect content. Ignore outliers that have nothing to do with the question. Use all search result of 3 languages to answer the question. Only answer what is asked. Answers should be short and concise. Don't add prior knowledge to the answer. Answer step by step. Answered by {lang}.

Query: {question}
Answer:
"""

####################################################################################

prompt_template_predict = """
Search result:
----------------------
      {context_str}
----------------------

Query: Summarize into bullet points (each bullet point is an important idea) the main ideas of the paragraph in {lang}.
Answer:
"""


####################################################################################