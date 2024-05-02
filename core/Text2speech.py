from core.util import *


def text_to_speech(answer, azure_speech_key, azure_speech_region, output_file):
    speech_config = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_speech_region)
    languages_first_question = detect_language_prompt(answer)
    if languages_first_question == "Korean":
        speech_config.speech_synthesis_voice_name = "ko-KR-YuJinNeural"
    elif languages_first_question == "Japanese":
        speech_config.speech_synthesis_voice_name = "ja-JP-ShioriNeural"
    elif languages_first_question == "English":
        speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    speech_synthesizer_result = speech_synthesizer.speak_text_async(answer).get()

    return output_file