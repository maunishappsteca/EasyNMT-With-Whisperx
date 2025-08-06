import runpod
from easynmt import EasyNMT
import fasttext
import nltk
import os

# Download nltk tokenizer
nltk.download('punkt')

# Load FastText language ID model
FASTTEXT_MODEL_PATH = "lid.176.ftz"
if not os.path.exists(FASTTEXT_MODEL_PATH):
    raise FileNotFoundError("FastText language model 'lid.176.ftz' not found.")

fasttext.FastText.eprint = lambda x: None  # Silence warnings
lang_detect_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

# Load EasyNMT with FastText language detection
model = EasyNMT('opus-mt', lang_detect_model=lang_detect_model)

# Handler for RunPod
def handler(job):
    try:
        input_data = job["input"]

        sentences = input_data.get("sentences")
        target_lang = input_data.get("target_lang", "en")
        source_lang = input_data.get("source_lang", None)  # May be None or "-"

        if not sentences:
            return {"error": "Missing 'sentences' input"}

        # If source_lang is "-", perform auto-detection using fasttext
        if source_lang == "-":
            detected_languages = [
                lang_detect_model.predict(sentence)[0][0].replace("__label__", "")
                for sentence in sentences
            ]

            # Translate each sentence individually using detected source_lang
            translations = []
            for sent, src_lang in zip(sentences, detected_languages):
                translated = model.translate(sent, source_lang=src_lang, target_lang=target_lang)
                translations.append(translated)

            return {
                "source_languages": detected_languages,
                "input": sentences,
                "translations": translations
            }

        else:
            # Use provided source_lang (or auto if it's None)
            translations = model.translate(sentences, source_lang=source_lang, target_lang=target_lang)
            return {
                "input": sentences,
                "translations": translations
            }

    except Exception as e:
        return {"error": str(e)}

# Start the RunPod serverless app
runpod.serverless.start({"handler": handler})
