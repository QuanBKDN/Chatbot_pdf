from core.Preproccesing import *
from core.util import *

def predict_file(inputs_first_question,channel,path_true,file_names):
    languages_first_question = detect_language_prompt(inputs_first_question)
    em_question = embed(inputs_first_question)
    embed_continue = []
    output_list = []
    data = load_embedding(channel) # Data la mot dict : {"key":"filename","value":"[[....]]"}
    list_trunks = get_chunks(path_true)
    if len(file_names) == 0:
        output_list = []
        thershold = 0
    elif len(file_names) > 0.5: 
        for filename in file_names:
            embed_loaded = data[filename]
            list_trunk = list_trunks[filename]
            similarity = np.inner(em_question, embed_loaded)
            max_value = np.array(similarity[0])
            max_list = max(similarity[0])
            max_index = np.argmax(max_value)
            max_ans = list_trunk[max_index]
            output_list.append(max_ans)
            embed_continue.append(np.array(max_list[0]))
        thershold = min((np.array(embed_continue)))
        
    return thershold,output_list,languages_first_question

def call_api_azureopenai(openai_key,inputs_first_question ,languages_first_question, files,thershold,checkfile):
    openai.api_key = openai_key
    if len(checkfile)==0 :
        prompt = prompt_template.replace("{question}", inputs_first_question).replace("{lang}", languages_first_question)
        completions = openai.ChatCompletion.create(
            engine="chatgpt-test2",
            max_tokens = 500,
            temperature=0.5,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ]
        )
        message_openai = completions.choices[0].message.content.lstrip("\n")
        prompt_tokens = completions['usage']['prompt_tokens']
        total_tokens = completions['usage']['total_tokens']

        return {"question": inputs_first_question, "answer_openai": message_openai,"answer_pdf": None, "redoc":None,
                "prompt_tokens":prompt_tokens,"total_tokens":total_tokens,"status":"success"}

    elif len(checkfile) > 0.5:
        if thershold < 0.03 :
            prompt = prompt_template.replace("{question}", inputs_first_question).replace("{lang}", languages_first_question)
            completions = openai.ChatCompletion.create(
            engine="chatgpt-test2",
            max_tokens = 500,
            temperature=0.5,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ]
        )
            message_openai = completions.choices[0].message.content.lstrip("\n")
            prompt_tokens = completions['usage']['prompt_tokens']
            total_tokens = completions['usage']['total_tokens']

            return {"question": inputs_first_question, "answer_openai": message_openai,"answer_pdf": None,"redoc":None,
                    "prompt_tokens":prompt_tokens,"total_tokens":total_tokens,"status":"success"}

        if files :
            if len(files) > 1:
                file_to_join = [c for c in files if c is not None]
                context_str = "\n\n".join(file_to_join)
                embed_content = embed(file_to_join)
                prompt = prompt_template.replace("{question}", inputs_first_question).replace(
                    "{context_str}", context_str).replace("{lang}", languages_first_question)
                completions = openai.ChatCompletion.create(
                    engine="chatgpt-test2",
                    temperature= 0.5,
                    max_tokens = 500,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": prompt},
                    ],
                )
                message_pdf = completions.choices[0].message.content.lstrip("\n")
                embed_message = embed(message_pdf)
                simmilarity_messagge = np.inner(embed_message,embed_content)
                max_value_mess = np.array(simmilarity_messagge[0])
                max_index_mess = np.argmax(max_value_mess)
                max_ans_mess = file_to_join[max_index_mess]
                if max(max_value_mess) < 0.4 : 
                    redoc_multi = None
                else:
                    redoc_multi = add_source_numbers([max_ans_mess])
                prompt_tokens = completions['usage']['prompt_tokens']
                total_tokens = completions['usage']['total_tokens']
                
                return {"question": inputs_first_question, "answer_openai": None,"answer_pdf": message_pdf,"redoc":redoc_multi,
                        "prompt_tokens":prompt_tokens,"total_tokens":total_tokens,"status":"success"}
            
            elif len(files) == 1: 
                redoc = add_source_numbers(files)
                prompt = prompt_template.replace("{question}", inputs_first_question).replace(
                    "{context_str}", files[0]).replace("{lang}", languages_first_question)
                completions = openai.ChatCompletion.create(
                    engine="chatgpt-test2",
                    temperature=0.5,
                    max_tokens = 500,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": prompt},
                    ],
                )
                message_pdf = completions.choices[0].message.content.lstrip("\n")
                prompt_tokens = completions['usage']['prompt_tokens']
                total_tokens = completions['usage']['total_tokens']

                return {"question": inputs_first_question, "answer_openai": None,"answer_pdf": message_pdf,"redoc":redoc, 
                        "prompt_tokens":prompt_tokens,"total_tokens":total_tokens,"status":"success"}