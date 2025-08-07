import runpod
from easynmt import EasyNMT
import fasttext
import nltk
import os
import logging
import sys
import signal
import numpy as np

# Configure numpy compatibility
np.fastCopy = False  # Workaround for NumPy 2.0 compatibility

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("🔄 Starting server...")  # stdout for RunPod logs

# Graceful shutdown handler
def handle_shutdown(signum, frame):
    logger.info("🚨 Received shutdown signal, exiting...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# Configure NLTK data path
try:
    nltk.data.path.append('/usr/share/nltk_data')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir='/usr/share/nltk_data')
    logger.info("✅ NLTK 'punkt' tokenizer ready")
except Exception as e:
    logger.error(f"❌ NLTK setup failed: {str(e)}")
    sys.exit(1)

# Load FastText language ID model
FASTTEXT_MODEL_PATH = "lid.176.bin"  # Using .bin version directly
if not os.path.exists(FASTTEXT_MODEL_PATH):
    logger.error(f"❌ FastText model not found at {FASTTEXT_MODEL_PATH}")
    sys.exit(1)

try:
    # Silence FastText warnings
    fasttext.FastText.eprint = lambda x: None
    logger.info("⏳ Loading FastText model...")
    lang_detect_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
    
    # Verify model works with test predictions
    test_pred_de = lang_detect_model.predict("Das ist ein Test", k=1)
    test_pred_fr = lang_detect_model.predict("C'est un test", k=1)
    if (not test_pred_de[0][0].startswith('__label__de') or 
        not test_pred_fr[0][0].startswith('__label__fr')):
        raise Exception("FastText model verification failed")
    
    logger.info("✅ FastText model loaded and verified successfully")
except Exception as e:
    logger.error(f"❌ FastText load failed: {str(e)}")
    sys.exit(1)

# Load translation model
try:
    logger.info("⏳ Loading EasyNMT model...")
    model = EasyNMT(
        'opus-mt',
        lang_detect_model=lang_detect_model,
        device='cpu',
        load_only=['opus-mt']
    )
    logger.info("✅ EasyNMT model loaded successfully")
except Exception as e:
    logger.error(f"❌ EasyNMT init failed: {str(e)}")
    sys.exit(1)

def handler(job):
    try:
        input_data = job.get("input", {})
        sentences = input_data.get("sentences")
        target_lang = input_data.get("target_lang", "en")
        source_lang = input_data.get("source_lang", None)

        if not sentences:
            return {"error": "No 'sentences' provided in input"}
        if not target_lang:
            return {"error": "No 'target_lang' provided in input"}

        # Ensure sentences is a list of strings
        if isinstance(sentences, str):
            sentences = [sentences]
        sentences = [str(s).strip() for s in sentences if str(s).strip()]

        if source_lang == "-":
            detected_languages = []
            for sentence in sentences:
                try:
                    # Get prediction with k=1 (top prediction only)
                    pred = lang_detect_model.predict(sentence, k=1)
                    if not pred[0] or not pred[0][0].startswith('__label__'):
                        raise ValueError("Invalid prediction format")
                    
                    lang_code = pred[0][0].replace("__label__", "")
                    logger.info(f"Detected language for '{sentence[:20]}...': {lang_code}")
                    detected_languages.append(lang_code)
                except Exception as e:
                    logger.error(f"Language detection failed for: '{sentence[:20]}...' Error: {str(e)}")
                    return {"error": f"Language detection failed: {str(e)}"}

            # Convert to numpy array with explicit dtype
            detected_languages_np = np.array(detected_languages, dtype='object')

            translations = model.translate(
                sentences,
                source_lang=detected_languages_np,
                target_lang=target_lang,
                batch_size=8
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
                batch_size=8
            )
            return {
                "input": sentences,
                "translations": translations
            }

    except Exception as e:
        logger.error(f"❌ Handler error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info("🚀 Starting RunPod handler")
    runpod.serverless.start({"handler": handler})