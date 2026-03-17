"""
Text Translation Service
Detects language and translates non-English text to English
for accurate emotion analysis with English-only models.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy-loaded translator
_translator = None
_langdetect_loaded = False

# Languages that should be translated (Bosnian, Croatian, Serbian, etc.)
SLAVIC_LANGUAGES = {'bs', 'hr', 'sr', 'mk', 'sl', 'bg', 'cs', 'sk', 'pl', 'uk', 'ru'}
# All non-English languages that we should translate
TRANSLATE_LANGUAGES = SLAVIC_LANGUAGES | {
    'de', 'fr', 'es', 'it', 'pt', 'nl', 'tr', 'ar', 'zh', 'ja', 'ko',
    'hi', 'bn', 'ro', 'hu', 'fi', 'sv', 'no', 'da', 'el', 'th', 'vi',
}


def _contains_bosnian_markers(text: str) -> bool:
    """
    Heuristic check: does the text contain Bosnian/Croatian/Serbian words?
    langdetect often misclassifies short BCS text as Indonesian, Somali, etc.
    """
    text_lower = text.lower()
    # Common BCS words that are unlikely in other languages
    bcs_markers = {
        'sam', 'nisam', 'jesam', 'imam', 'nemam', 'nista', 'ništa',
        'kako', 'znam', 'mogu', 'hocu', 'hoću', 'necu', 'neću',
        'danas', 'sutra', 'jucer', 'jučer', 'sada', 'jako', 'veoma',
        'ali', 'jer', 'nego', 'ili', 'kad', 'kada', 'gdje', 'sto', 'što',
        'osjecam', 'osjećam', 'mislim', 'zelim', 'želim', 'trebam',
        'tuzan', 'tužan', 'sretan', 'sretno', 'ljut', 'ljuta',
        'uplašen', 'uplasen', 'bojim', 'plasim', 'plašim',
        'volim', 'mrzim', 'zivot', 'život', 'dobro', 'loše', 'lose',
        'niko', 'neko', 'sve', 'uvijek', 'nikad', 'uvijek',
        'mi', 'ide', 'treba', 'moze', 'može', 'mora',
        'iznenadjen', 'iznenađen', 'zbunjen', 'umoran',
    }
    words = set(text_lower.replace(',', ' ').replace('.', ' ').replace('!', ' ').split())
    overlap = words & bcs_markers
    # If 2+ BCS marker words found, it's likely Bosnian/Croatian/Serbian
    return len(overlap) >= 2


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    Returns ISO 639-1 code (e.g. 'en', 'bs', 'hr', 'sr').
    Includes heuristic for Bosnian/Croatian/Serbian which langdetect often misclassifies.
    """
    global _langdetect_loaded

    # First: heuristic check for Bosnian/Croatian/Serbian
    # langdetect is unreliable for short BCS text (often returns 'id', 'so', 'tl')
    if _contains_bosnian_markers(text):
        logger.info("BCS heuristic matched (%d chars)", len(text))
        return "hr"  # Use Croatian (Google Translate handles bs/hr/sr similarly)

    try:
        from langdetect import detect, DetectorFactory
        if not _langdetect_loaded:
            DetectorFactory.seed = 0  # Deterministic results
            _langdetect_loaded = True

        detected = detect(text)

        # If langdetect returns an unusual language for what might be BCS,
        # check for diacritical characters common in South Slavic
        if detected in ('id', 'so', 'tl', 'sw', 'ms', 'cy') and any(
            c in text.lower() for c in 'čćžšđ'
        ):
            logger.info("Overriding langdetect '%s' -> 'hr' due to diacritics", detected)
            return "hr"

        return detected
    except Exception as e:
        logger.warning("Language detection failed: %s", e)
        return "en"  # Default to English


def translate_to_english(text: str, source_lang: str = 'auto') -> Tuple[str, str, bool]:
    """
    Translate text to English if it's not already English.

    Args:
        text: Input text
        source_lang: Source language code, or 'auto' to detect

    Returns:
        Tuple of (translated_text, detected_language, was_translated)
    """
    if not text or not text.strip():
        return text, "en", False

    # Detect language
    if source_lang == 'auto':
        detected_lang = detect_language(text)
    else:
        detected_lang = source_lang

    # If English, no translation needed
    if detected_lang == 'en':
        return text, 'en', False

    # Try translation
    try:
        translated = _do_translate(text, detected_lang)
        if translated and translated.strip():
            logger.info("Translated [%s -> en] (%d chars -> %d chars)", detected_lang, len(text), len(translated))
            return translated, detected_lang, True
        else:
            return text, detected_lang, False
    except Exception as e:
        logger.warning("Translation failed for [%s]: %s", detected_lang, e)
        # Fallback: try keyword-based translation for common Bosnian phrases
        fallback = _keyword_translate(text)
        if fallback != text:
            return fallback, detected_lang, True
        return text, detected_lang, False


def _do_translate(text: str, source_lang: str) -> Optional[str]:
    """Perform the actual translation using deep-translator."""
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source=source_lang, target='en')
        return translator.translate(text)
    except ImportError:
        logger.warning("deep-translator not installed, trying googletrans")
        try:
            from googletrans import Translator
            global _translator
            if _translator is None:
                _translator = Translator()
            result = _translator.translate(text, src=source_lang, dest='en')
            return result.text
        except ImportError:
            logger.error("No translation library available")
            return None
    except Exception as e:
        logger.error("Translation error: %s", e)
        return None


# Common Bosnian/Croatian/Serbian emotion words -> English
_KEYWORD_MAP = {
    # Joy
    'sretno': 'happy', 'sretan': 'happy', 'sretna': 'happy',
    'veselo': 'cheerful', 'veseo': 'cheerful', 'vesela': 'cheerful',
    'radostan': 'joyful', 'radosna': 'joyful', 'radost': 'joy',
    'zadovoljan': 'satisfied', 'zadovoljna': 'satisfied',
    'odlično': 'excellent', 'odlicno': 'excellent',
    'super': 'super', 'divno': 'wonderful', 'prekrasno': 'beautiful',
    'volim': 'love', 'ljubav': 'love',
    'srećan': 'happy', 'srecan': 'happy', 'srećna': 'happy', 'srecna': 'happy',
    # Sadness
    'tužno': 'sadly', 'tužan': 'sad', 'tužna': 'sad',
    'tuzno': 'sadly', 'tuzan': 'sad', 'tuzna': 'sad',
    'žalosno': 'sorrowful', 'žalostan': 'sad', 'žalosna': 'sad',
    'zalosno': 'sorrowful', 'zalostan': 'sad', 'zalosna': 'sad',
    'plačem': 'crying', 'placem': 'crying', 'suze': 'tears',
    'usamljen': 'lonely', 'usamljena': 'lonely',
    'depresivan': 'depressed', 'depresivna': 'depressed',
    'loše': 'bad', 'lose': 'bad', 'nesretan': 'unhappy', 'nesretna': 'unhappy',
    # Anger
    'ljut': 'angry', 'ljuta': 'angry', 'ljutit': 'angry',
    'bijesan': 'furious', 'bijesna': 'furious',
    'besan': 'furious', 'besna': 'furious',
    'frustriran': 'frustrated', 'frustrirana': 'frustrated',
    'nervozan': 'nervous', 'nervozna': 'nervous',
    'mrzim': 'hate', 'iritiran': 'irritated',
    # Fear
    'uplašen': 'scared', 'uplašena': 'scared',
    'uplasen': 'scared', 'uplasena': 'scared',
    'strah': 'fear', 'bojim': 'afraid', 'plašim': 'afraid', 'plasim': 'afraid',
    'zabrinut': 'worried', 'zabrinuta': 'worried',
    'anksiozan': 'anxious', 'anksiozna': 'anxious',
    'prestrašen': 'terrified', 'prestrasen': 'terrified',
    # Surprise
    'iznenađen': 'surprised', 'iznenadjen': 'surprised',
    'iznenađena': 'surprised', 'iznenadjena': 'surprised',
    'šokiran': 'shocked', 'sokiran': 'shocked',
    'začuđen': 'amazed', 'zacudjen': 'amazed',
    'nevjerica': 'disbelief', 'nevjerovatno': 'unbelievable',
    # Disgust
    'gadi': 'disgusted', 'gadljivo': 'disgusting',
    'odvratno': 'disgusting', 'užasno': 'horrible', 'uzasno': 'horrible',
    'zgađen': 'disgusted', 'zgadjen': 'disgusted',
    # Neutral / General
    'osjećam': 'feel', 'osjecam': 'feel',
    'dobro': 'well', 'okej': 'okay', 'normalno': 'normal',
    'umoran': 'tired', 'umorna': 'tired',
    'zbunjen': 'confused', 'zbunjena': 'confused',
    'smiren': 'calm', 'smirena': 'calm', 'mirno': 'calmly',
    'uzbuđen': 'excited', 'uzbudjen': 'excited',
    'uzbuđena': 'excited', 'uzbudjena': 'excited',
    # Common verbs/phrases
    'nisam': 'am not', 'jesam': 'am', 'imam': 'have',
    'danas': 'today', 'sada': 'now', 'jako': 'very',
    'malo': 'a little', 'mnogo': 'a lot', 'veoma': 'very',
    'mi': 'me', 'se': 'self', 'sam': 'am',
}


def _keyword_translate(text: str) -> str:
    """
    Fallback: word-by-word translation using keyword map.
    Used when online translation is unavailable.
    """
    words = text.lower().split()
    translated_words = []
    any_translated = False

    for word in words:
        # Strip common punctuation
        clean = word.strip('.,!?;:()[]{}"\'-')
        if clean in _KEYWORD_MAP:
            translated_words.append(_KEYWORD_MAP[clean])
            any_translated = True
        else:
            translated_words.append(word)

    if any_translated:
        return ' '.join(translated_words)
    return text
