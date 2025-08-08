import os
import uuid
import subprocess
import whisperx
import runpod
import boto3
import gc
import json
import nltk
import logging
import sys
import fasttext
import numpy as np
from typing import Optional
from botocore.exceptions import ClientError
from datetime import datetime
from easynmt import EasyNMT

# --- Configuration ---
COMPUTE_TYPE = "float16"
BATCH_SIZE = 16
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
MODEL_CACHE_DIR = os.getenv("WHISPER_MODEL_CACHE", "/app/models")
TRANSLATION_MODEL_DIR = "/app/translation_models"
WORD_TRANSLATION_ENABLED = True  # Control word-level translations

# Configure numpy compatibility
np.fastCopy = False  # Workaround for NumPy 2.0 compatibility

# Configure logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Initialize S3 client
s3 = boto3.client('s3') if S3_BUCKET else None

# --- Translation Model Setup ---
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
FASTTEXT_MODEL_PATH = "lid.176.bin"
if not os.path.exists(FASTTEXT_MODEL_PATH):
    logger.error(f"❌ FastText model not found at {FASTTEXT_MODEL_PATH}")
    sys.exit(1)

try:
    # Silence FastText warnings
    fasttext.FastText.eprint = lambda x: None
    logger.info("⏳ Loading FastText model...")
    lang_detect_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
    logger.info("✅ FastText model loaded successfully")
except Exception as e:
    logger.error(f"❌ FastText load failed: {str(e)}")
    sys.exit(1)

# Load translation model
try:
    logger.info("⏳ Loading EasyNMT model...")
    translation_model = EasyNMT(
        'opus-mt',
        lang_detect_model=lang_detect_model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        load_only=['opus-mt'],
        cache_folder=TRANSLATION_MODEL_DIR
    )
    logger.info("✅ EasyNMT model loaded successfully")
except Exception as e:
    logger.error(f"❌ EasyNMT init failed: {str(e)}")
    sys.exit(1)

# --- Core Functions ---
def ensure_model_cache_dir():
    """Ensure model cache directory exists and is accessible"""
    try:
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        test_file = os.path.join(MODEL_CACHE_DIR, "test.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except Exception as e:
        logger.error(f"Model cache directory error: {str(e)}")
        return False

def convert_to_wav(input_path: str) -> str:
    """Convert media file to 16kHz mono WAV"""
    try:
        output_path = f"/tmp/{uuid.uuid4()}.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le",
            "-loglevel", "error",
            output_path
        ], check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed error: {str(e)}")
        raise RuntimeError(f"FFmpeg conversion failed: {str(e)}")
    except Exception as e:
        logger.error(f"Audio conversion error: {str(e)}")
        raise RuntimeError(f"Audio conversion error: {str(e)}")

def load_model(model_size: str, language: Optional[str]):
    """Load Whisper model with GPU optimization"""
    try:
        if not ensure_model_cache_dir():
            logger.error(f"Model cache directory is not accessible")
            raise RuntimeError("Model cache directory is not accessible")
            
        return whisperx.load_model(
            model_size,
            device="cuda",
            compute_type=COMPUTE_TYPE,
            download_root=MODEL_CACHE_DIR,
            language=language if language and language != "-" else None
        )
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

def load_alignment_model(language_code: str):
    """Load alignment model with fallback options"""
    try:
        return whisperx.load_align_model(language_code=language_code, device="cuda")
    except Exception as e:
        logger.warning(f"Failed to load default alignment model for {language_code}, trying fallback: {str(e)}")
        
        fallback_models = {
            "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
            "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
            "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
        }
        
        if language_code in fallback_models:
            try:
                return whisperx.load_align_model(
                    model_name=fallback_models[language_code],
                    device="cuda"
                )
            except Exception as fallback_e:
                logger.error(f"Failed to load fallback alignment model for {language_code}: {str(fallback_e)}")
                raise RuntimeError(f"Alignment model loading failed for {language_code}")
        else:
            logger.error(f"No alignment model available for language: {language_code}")
            raise RuntimeError(f"No alignment model available for language: {language_code}")

def translate_text(text: str, target_lang: str, source_lang: str = None):
    """Translate text using EasyNMT"""
    if not text or target_lang == "-":
        return None
    
    try:
        # Handle auto-detection if source language is not provided
        if source_lang in (None, "-", "auto"):
            translation = translation_model.translate(text, target_lang=target_lang)
        else:
            translation = translation_model.translate(text, 
                                                     source_lang=source_lang, 
                                                     target_lang=target_lang)
        return {
            "text": translation,
            "src_lang": source_lang
        }
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        return None

def translate_words(words: list, target_lang: str, source_lang: str):
    """Translate individual words with context awareness"""
    if not words or not WORD_TRANSLATION_ENABLED:
        return None
    
    try:
        # Extract word texts
        word_texts = [w["word"] for w in words]
        
        # Get translations in batch
        translations = translation_model.translate(
            word_texts,
            source_lang=source_lang,
            target_lang=target_lang,
            batch_size=32  # Larger batch for words
        )
        
        return translations
    except Exception as e:
        logger.warning(f"Word translation failed, continuing without word-level translations: {str(e)}")
        return None

def translate_segments(segments: list, target_lang: str, source_lang: str):
    """Translate segments and words with proper error handling"""
    if not segments or target_lang == "-" or target_lang == source_lang:
        return format_segments(segments)
    
    try:
        # First translate all segment texts
        segment_texts = [seg["text"] for seg in segments]
        translated_texts = translation_model.translate(
            segment_texts,
            source_lang=source_lang,
            target_lang=target_lang,
            batch_size=8
        )
        
        translated_segments = []
        for i, seg in enumerate(segments):
            # Handle word translations if enabled and words exist
            word_translations = None
            if WORD_TRANSLATION_ENABLED and "words" in seg:
                word_translations = translate_words(seg["words"], target_lang, source_lang)
            
            # Build words array
            words = []
            if "words" in seg:
                for j, word in enumerate(seg["words"]):
                    trans = word_translations[j] if word_translations and j < len(word_translations) else None
                    words.append({
                        **word,
                        "word_translation": trans
                    })
            
            translated_segments.append({
                **seg,
                "text_translation": translated_texts[i],
                "words": words if words else seg.get("words", [])
            })
        
        return translated_segments
    except Exception as e:
        logger.error(f"Segment translation failed: {str(e)}")
        return format_segments(segments)

def format_segments(segments: list):
    """Ensure consistent segment structure"""
    if not segments:
        return segments

    formatted_segments = []
    for segment in segments:
        words = []
        if "words" in segment:
            for word in segment["words"]:
                words.append({
                    **word,
                    "word_translation": word.get("word_translation")  # Preserve existing if present
                })
                
        formatted_segments.append({
            **segment,
            "text_translation": segment.get("text_translation"),
            "words": words
        })

    return formatted_segments

def transcribe_audio(audio_path: str, model_size: str, language: Optional[str], align: bool, translate_to: Optional[str]):
    """Core transcription logic with robust error handling"""
    try:
        model = load_model(model_size, language)
        result = model.transcribe(audio_path, batch_size=BATCH_SIZE)
        detected_language = result.get("language", language if language else "en")
        
        if align and detected_language != "unknown":
            try:
                align_model, metadata = load_alignment_model(detected_language)
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    metadata,
                    audio_path,
                    device="cuda",
                    return_char_alignments=False
                )
            except Exception as e:
                logger.error(f"Alignment skipped: {str(e)}")
                result["alignment_error"] = str(e)
        
        # Handle translation if requested
        translated_text = None
        translated_segments = None
        
        if translate_to and translate_to != "-":
            try:
                full_text = " ".join(seg["text"] for seg in result["segments"])
                translation_result = translate_text(full_text, translate_to, detected_language)
                if translation_result:
                    translated_text = translation_result["text"]
                
                translated_segments = translate_segments(result["segments"], translate_to, detected_language)
            except Exception as e:
                logger.error(f"Translation failed, returning untranslated text: {str(e)}")
                translated_segments = format_segments(result["segments"])
        else:
            translated_segments = format_segments(result["segments"])

        return {
            "text": " ".join(seg["text"] for seg in result["segments"]),
            "translation": translated_text,
            "segments": translated_segments if translated_segments else result["segments"],
            "language": detected_language,
            "model": model_size,
            "alignment_success": "alignment_error" not in result,
            "word_translations_included": WORD_TRANSLATION_ENABLED and translate_to and translate_to != "-"
        }
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise RuntimeError(f"Transcription failed: {str(e)}")

def handler(job):
    """RunPod serverless handler with comprehensive error handling"""
    try:
        if not job.get("input"):
            return {"error": "No input provided"}
            
        input_data = job["input"]
        file_name = input_data.get("file_name")
        
        if not file_name:
            return {"error": "No file_name provided in input"}
        
        # 1. Download from S3
        local_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_name)}"
        try:
            if S3_BUCKET:
                s3.download_file(S3_BUCKET, file_name, local_path)
            else:
                return {"error": "S3 bucket not configured"}
        except Exception as e:
            return {"error": f"S3 download failed: {str(e)}"}
        
        # 2. Convert to WAV if needed
        try:
            if not file_name.lower().endswith('.wav'):
                audio_path = convert_to_wav(local_path)
                os.remove(local_path)
            else:
                audio_path = local_path
        except Exception as e:
            if os.path.exists(local_path):
                os.remove(local_path)
            return {"error": f"Audio processing failed: {str(e)}"}
        
        # 3. Transcribe
        try:
            result = transcribe_audio(
                audio_path,
                input_data.get("model_size", "large-v3"),
                input_data.get("language", None),
                input_data.get("align", False),
                input_data.get("translateTo", "-")
            )
        except Exception as e:
            return {"error": str(e)}
        finally:
            # 4. Cleanup
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                gc.collect()
            except Exception as e:
                logger.warning(f"Cleanup failed: {str(e)}")
        
        return result
        
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

if __name__ == "__main__":
    print("Starting WhisperX cuda Endpoint with Enhanced EasyNMT Translation...")
    
    # Verify model cache directory at startup
    if not ensure_model_cache_dir():
        print("ERROR: Model cache directory is not accessible")
        if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
            raise RuntimeError("Model cache directory is not accessible")
    
    if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
        runpod.serverless.start({"handler": handler})
    else:
        # Test with mock input
        test_result = handler({
            "input": {
                "file_name": "test.wav",
                "model_size": "base",
                "language": "hi",
                "translateTo": "en",
                "align": True,
                "word_level_translation": True
            }
        })
        print("Test Result:", json.dumps(test_result, indent=2))