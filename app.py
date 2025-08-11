import os
import uuid
import subprocess
import whisperx
import runpod
import boto3
import gc
import json
import logging
import sys
import torch
from typing import Optional
from botocore.exceptions import ClientError
from easynmt import EasyNMT

# --- Configuration ---
COMPUTE_TYPE = "float16"
BATCH_SIZE = 16
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
MODEL_CACHE_DIR = os.getenv("WHISPER_MODEL_CACHE", "/app/models")
TRANSLATION_MODEL_DIR = "/app/translation_models"

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

# Load translation model
try:
    logger.info("⏳ Loading EasyNMT model...")
    translation_model = EasyNMT(
        'opus-mt',
        device='cuda',
        load_only=['opus-mt'],
        cache_folder=TRANSLATION_MODEL_DIR
    )
    logger.info("✅ EasyNMT model loaded successfully")
    logger.info(f"Supported languages: {translation_model.supported_languages}")
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

def clear_model_cache():
    """Clear non-essential files from model cache"""
    try:
        logger.info("🧹 Cleaning up model cache...")
        keep_extensions = ('.bin', '.pt', '.pth', '.model', '.json', '.txt')
        for root, dirs, files in os.walk(MODEL_CACHE_DIR):
            for file in files:
                if not file.endswith(keep_extensions):
                    try:
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
                        logger.debug(f"Removed cache file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Couldn't remove {file}: {str(e)}")
        logger.info("✅ Model cache cleaned")
    except Exception as e:
        logger.warning(f"Model cache cleanup failed: {str(e)}")

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

def translate_text(text: str, target_lang: str, source_lang: str):
    """Translate text using EasyNMT"""
    if not text or target_lang == "-":
        return None
    
    try:
        # Verify languages are supported
        if source_lang not in translation_model.supported_languages:
            raise RuntimeError(f"Source language {source_lang} not supported")
        if target_lang not in translation_model.supported_languages:
            raise RuntimeError(f"Target language {target_lang} not supported")
            
        return translation_model.translate(text, source_lang=source_lang, target_lang=target_lang)
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        return None

def translate_words(words: list, target_lang: str, source_lang: str):
    """Translate individual words with proper batching"""
    if not words:
        return None
    
    try:
        # Verify languages are supported
        if source_lang not in translation_model.supported_languages:
            raise RuntimeError(f"Source language {source_lang} not supported")
        if target_lang not in translation_model.supported_languages:
            raise RuntimeError(f"Target language {target_lang} not supported")
            
        # Extract word texts
        word_texts = [w["word"] for w in words]
        
        # Get translations in batch
        translations = translation_model.translate(
            word_texts,
            source_lang=source_lang,
            target_lang=target_lang,
            batch_size=32
        )
        
        return translations
    except Exception as e:
        logger.error(f"Word translation failed: {str(e)}")
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
            # Translate words if available
            word_translations = None
            if "words" in seg and seg["words"]:
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
                    "word_translation": None
                })
                
        formatted_segments.append({
            **segment,
            "text_translation": None,
            "words": words
        })

    return formatted_segments

def transcribe_audio(audio_path: str, model_size: str, language: Optional[str], align: bool, translate_to: Optional[str]):
    """Core transcription logic with robust error handling"""
    try:
        model = load_model(model_size, language)
        result = model.transcribe(audio_path, batch_size=BATCH_SIZE)
        detected_language = result.get("language", language if language and language != "-" else None)
        
        # If no language specified and nothing detected, default to English
        if not detected_language:
            detected_language = "en"
            logger.warning("No language detected, defaulting to English")

        # Improved language detection verification
        if language == "-" and result.get("segments"):
            first_segment = result["segments"][0]
            if "language_probability" in first_segment:
                if first_segment["language_probability"] < 0.5:  # Low confidence threshold
                    logger.warning(f"Low detection confidence ({first_segment['language_probability']:.2f}) for {detected_language}")
                    # Try forcing English if detection is uncertain
                    detected_language = "en"
                    logger.info("Overriding detection with English due to low confidence")

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
        
        if translate_to and translate_to != "-" and translate_to != detected_language:
            try:
                full_text = " ".join(seg["text"] for seg in result["segments"])
                translated_text = translate_text(full_text, translate_to, detected_language)
                translated_segments = translate_segments(result["segments"], translate_to, detected_language)
                
                if translated_text is None:
                    raise RuntimeError("Translation returned None")
                    
            except Exception as e:
                logger.error(f"Translation failed: {str(e)}")
                translated_segments = format_segments(result["segments"])
        else:
            translated_segments = format_segments(result["segments"])

        return {
            "text": " ".join(seg["text"] for seg in result["segments"]),
            "translation": translated_text,
            "segments": translated_segments,
            "language": detected_language,
            "model": model_size,
            "alignment_success": "alignment_error" not in result,
            "word_translations_included": translate_to and translate_to != "-" and translated_text is not None and "words" in result["segments"][0],
            "detection_confidence": result["segments"][0].get("language_probability", None) if result.get("segments") else None
        }
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise RuntimeError(f"Transcription failed: {str(e)}")
    finally:
        # Clear memory after each transcription
        torch.cuda.empty_cache()
        gc.collect()

def handler(job):
    """RunPod serverless handler with comprehensive error handling"""
    files_to_cleanup = []
    try:
        if not job.get("input"):
            return {"error": "No input provided"}
            
        input_data = job["input"]
        file_name = input_data.get("file_name")
        
        if not file_name:
            return {"error": "No file_name provided in input"}
        
        # 1. Download from S3
        local_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_name)}"
        files_to_cleanup.append(local_path)
        try:
            if S3_BUCKET:
                s3.download_file(S3_BUCKET, file_name, local_path)
                logger.info(f"Downloaded file from S3: {file_name}")
            else:
                return {"error": "S3 bucket not configured"}
        except Exception as e:
            return {"error": f"S3 download failed: {str(e)}"}
        
        # 2. Convert to WAV if needed
        try:
            if not file_name.lower().endswith('.wav'):
                audio_path = convert_to_wav(local_path)
                files_to_cleanup.append(audio_path)
                logger.info(f"Converted to WAV: {audio_path}")
            else:
                audio_path = local_path
        except Exception as e:
            return {"error": f"Audio processing failed: {str(e)}"}
        
        # 3. Transcribe
        try:
            logger.info("Starting transcription...")
            result = transcribe_audio(
                audio_path,
                input_data.get("model_size", "large-v3"),
                input_data.get("language", None),
                input_data.get("align", False),
                input_data.get("translateTo", "-")
            )
            logger.info("Transcription completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return {"error": str(e)}
            
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}
    finally:
        # 4. Comprehensive Cleanup
        logger.info("Starting cleanup...")
        cleanup_errors = 0
        
        # Cleanup all temporary files
        for file_path in files_to_cleanup:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Cleaned up file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup file {file_path}: {str(e)}")
                cleanup_errors += 1
        
        # Clear model cache of temporary files (keeps models)
        clear_model_cache()
        
        # GPU and memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        if cleanup_errors > 0:
            logger.warning(f"Cleanup completed with {cleanup_errors} errors")
        else:
            logger.info("Cleanup completed successfully")

if __name__ == "__main__":
    print("Starting WhisperX cuda Endpoint with Full Translation...")
    
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
                "language": "-",  # Auto-detect
                "translateTo": "en",
                "align": True
            }
        })
        print("Test Result:", json.dumps(test_result, indent=2))
