from core.util import *

def summary(path_file):
    text_file = pdf_to_text(path_file)
    language_text_file = detect_language_pdf(text_file)
    files_to_join = [c for c in text_file if c is not None]
    context = "\n\n".join(files_to_join)

    return context,language_text_file