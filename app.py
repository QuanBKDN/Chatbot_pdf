from flask import Flask, request, jsonify, send_file
import azure.cognitiveservices.speech as speechsdk
from flask_cors import CORS
from pydub import AudioSegment

from core.logger import logger
from core.util import *
from core.config import *
from core.Text2speech import *
from core.Deployment import *
from core.Speech2text import *
from core.Text2speech import *
from core.Text_summarize import *

app = Flask(__name__)
CORS(app)


#textospeech
@app.route('/api/v1/texttospeech', methods=['POST'])
def texttospeech():
    try:

        data = request.get_json()
        answer = data.get("answer")
        azure_speech_key = data.get("azure_speech_key")
        azure_speech_region = data.get("azure_speech_region")
        output_file = "output.wav"
        result = text_to_speech(answer, azure_speech_key, azure_speech_region, output_file)

        return send_file(result, as_attachment=True, download_name="output.wav",)

    except Exception as e:

        error_response = {
            "status": "fail",
            "message": str(e)
        }
        return jsonify(error_response), 500

#speechtotext
@app.route('/api/v1/recognizespeech', methods=['POST'])
def recognizespeech():
    try:

        json_data = request.json
        logger.debug(json_data)
        wav_file_path = json_data.get("wav_file_path")
        azure_speech_key = json_data.get("azure_speech_key")
        azure_speech_region = json_data.get("azure_speech_region")
        convert_to_wav(wav_file_path)
        result_text = recognize_speech_from_wav(wav_file_path, azure_speech_key, azure_speech_region)

        return {"recognized_text": result_text,"status": "success"} ,200

    except Exception as e:

        error_response = {
            "status": "fail",
            "message": str(e)
        }

        return jsonify(error_response), 500

#upload_file
@app.route('/api/v1/upload', methods=['POST'])
def upload():
    try:
        json_data = request.json
        # app.logger.debug(json_data)
        channel = json_data.get("channel")
        data_path = os.path.join(channel, 'data')
        embedding_data_path = os.path.join(channel, 'embedding_data')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if not os.path.exists(embedding_data_path):
            os.makedirs(embedding_data_path)
        upload_file(data_path,embedding_data_path)

        return {"status": "success"}, 200

    except Exception as e:

        error_response = {
                "status": "fail",
                "message": str(e)
            }

        return jsonify(error_response), 500

#delete_file
@app.route("/api/v1/delete", methods=["POST"])
def delete_files():
    response = {"status": None}
    json = request.json
    list_path_file = json.get("list_path_file")
    try:
        for file in list_path_file:
            delete_npy = delete_file_npy(file)
            os.remove(file)
        response["status"] = delete_npy

        return response

    except Exception as e:
        error_response = {
                "status": "fail",
                "message": str(e)
            }

        return jsonify(error_response), 500

#predict
@app.route('/api/v1/predict', methods=['POST'])
def predict() :
    try:
        json = request.json
        logger.debug(json)
        inputs = json.get("prompt")
        channel = json.get("channel")
        files = json.get("filename")
        file_names = [file["filename"] for file in json["files"]]
        openai_key = json.get("openai_key")
        app.logger.debug(inputs)
        path = json.get("channel")
        path_true = os.path.join(path,'data')
        if not os.path.exists(path_true):
            os.makedirs(path_true)
        checkfile = os.listdir(path_true)
        thershold, max_ans, languages_first_question = predict_file(inputs,channel, path_true, file_names)
        logger.debug(thershold)        
        result = call_api_azureopenai(openai_key,inputs, languages_first_question, max_ans, thershold, checkfile)

        return result
    
    except Exception as e:

        error_response = {
                "status": "fail",
                "message": str(e)
            }

        return jsonify(error_response), 500

#text_summarize
@app.route("/api/v1/summarize", methods=["POST"])
def textsummarize():
    try:
        json = request.json
        logger.debug(json)
        path_file = json.get("path_file")
        openai_key = json.get("openai_key")
        input_text = pdf_to_text(path_file)
        language_text_file = detect_language_pdf(input_text)
        openai.api_key = openai_key
        prompt_for_predict = prompt_template_predict.replace("{context_str}", input_text[0]).replace("{lang}", language_text_file)
        completion_for_predict = openai.ChatCompletion.create(
                engine="chatgpt-test2",
                temperature=0.5,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt_for_predict},
                ]
            )
        message_summary = completion_for_predict.choices[0].message.content.lstrip("\n")

        return {"summary": message_summary,
                "status" : "success"
        }
        
    except Exception as e:
        error_response = {
            "status": "fail",
            "message": str(e)
        }

        return jsonify(error_response),500

if __name__ == '__main__':
    logger.debug("APIs are online")
    app.run(debug=True, host=HOST, port=5000)
