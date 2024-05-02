import os
import re

from core.util import *


def upload_file(data_path,embedding_data_path):
    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_path, filename)
            # Process the PDF file
            check_pdf = pdf_to_text(file_path)
            chunks = text_to_chunks(check_pdf, file_path)
            check_language = detect_language_pdf(check_pdf)
            if check_language =='ja':
                list_ja_embed = []
                for chunk in chunks:
                    chunk_ja_embed = embed(chunk)
                    list_ja_embed.append(chunk_ja_embed)
                file_npy = re.sub(r'\.[^.]*$', '', filename)
                embedding_ja_path = os.path.join(embedding_data_path, f"{file_npy}.npy")
                np.save(embedding_ja_path,list_ja_embed)
            elif check_language =='en':
                list_en_embed =[]
                for chunk in chunks:
                    chunk_en_embed = embed(chunk)
                    list_en_embed.append(chunk_en_embed)
                file_npy = re.sub(r'\.[^.]*$', '', filename)
                embedding_en_path = os.path.join(embedding_data_path, f"{file_npy}.npy")
                np.save(embedding_en_path,list_en_embed)
            elif check_language =='ko':
                list_ko_embed =[]
                for chunk in chunks:
                    chunk_ko_embed = embed(chunk)
                    print(np.array(chunk_ko_embed).shape)
                    list_ko_embed.append(chunk_ko_embed)
                file_npy = re.sub(r'\.[^.]*$', '', filename)
                embedding_ko_path = os.path.join(embedding_data_path, f"{file_npy}.npy")
                np.save(embedding_ko_path,list_ko_embed)
                
                
def load_embedding(channel_path):
    embedding_folder = os.path.join(channel_path, 'embedding_data')
    if os.path.exists(embedding_folder):
        npy_files = [file for file in os.listdir(embedding_folder) if file.endswith('.npy')]
        embedding_dict = {}
        for npy_file in npy_files:
            npy_path = os.path.join(embedding_folder, npy_file)
            key = os.path.splitext(npy_file)[0] # Lấy tên tệp làm key (loại bỏ phần mở rộng .npy)
            embedding_dict[key] = np.load(npy_path)
        if embedding_dict:
            return embedding_dict
        else:
            return None
        
