import runpod
from easynmt import EasyNMT
import fasttext
import nltk
import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download nltk tokenizer
try:
    nltk.download('punkt', quiet=True)
    logger.info("NLTK punkt downloaded")
except Exception as e:
    logger.error(f"NLTK download failed: {str(e)}")
    sys.exit(1)

# Load FastText language ID model
FASTTEXT_MODEL_PATH = "lid.176.ftz"
if not os.path.exists(FASTTEXT_MODEL_PATH):
    logger.error(f"FastText model not found at {FASTTEXT_MODEL_PATH}")
    sys.exit(1)

try:
    # Silence FastText warnings
    fasttext.FastText.eprint = lambda x: None
    logger.info("Loading FastText model...")
    lang_detect_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
    logger.info("FastText model loaded successfully")
except Exception as e:
    logger.error(f"FastText load failed: {str(e)}")
    sys.exit(1)

# Load translation model with explicit CPU device
try:
    logger.info("Loading EasyNMT model...")
    model = EasyNMT(
        'opus-mt',
        lang_detect_model=lang_detect_model,
        device='cpu',  # Force CPU usage
        load_only=['opus-mt']  # Only load core model
    )
    logger.info("EasyNMT model loaded successfully")
except Exception as e:
    logger.error(f"EasyNMT init failed: {str(e)}")
    sys.exit(1)

# Handler for RunPod
def handler(job):
    try:
        input_data = job["input"]
        sentences = input_data.get("sentences")
        target_lang = input_data.get("target_lang", "en")
        source_lang = input_data.get("source_lang", None)

        if not sentences:
            return {"error": "Missing 'sentences' input"}

        # Handle auto-detection
        if source_lang == "-":
            # Batch detect languages
            detected_languages = []
            for sentence in sentences:
                pred = lang_detect_model.predict(sentence)
                lang_code = pred[0][0].replace("__label__", "")
                detected_languages.append(lang_code)
            
            # Batch translate
            translations = model.translate(
                sentences, 
                source_lang=detected_languages,
                target_lang=target_lang,
                batch_size=8  # Reduce batch size
            )
            
            return {
                "source_languages": detected_languages,
                "input": sentences,
                "translations": translations
            }
        else:
            translations = model.translate(
                sentences, 
                source_lang=source_lang, 
                target_lang=target_lang,
                batch_size=8  # Reduce batch size
            )
            return {
                "input": sentences,
                "translations": translations
            }

    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {"error": str(e)}

# Start the serverless app
if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler")
    runpod.serverless.start({"handler": handler})