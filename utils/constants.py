
from nltk.corpus import stopwords
import re
from nltk.stem import LancasterStemmer, SnowballStemmer

language_map = {
    "el": "greek",
    "pt": "portuguese",
    "ca": "catalan",
    "tl": "tagalog",
    "da": "danish",
    "hu": "hungarian",
    "ht": "haitian creole",
    "fr": "french",
    "qht": "haitian creole (Twitter-specific)",
    "is": "icelandic",
    "th": "thai",
    "pa": "punjabi",
    "am": "amharic",
    "und": "undetermined",
    "qst": "spanish (Twitter-specific)",
    "bn": "bengali",
    "en": "english",
    "cs": "czech",
    "sl": "slovene",
    "ro": "romanian",
    "eu": "basque",
    "vi": "vietnamese",
    "fi": "finnish",
    "ur": "urdu",
    "sv": "swedish",
    "cy": "welsh",
    "nl": "dutch",
    "qme": "meitei (Twitter-specific)",
    "it": "italian",
    "iw": "hebrew",  # Deprecated, use 'he' in modern standards
    "ta": "tamil",
    "zh": "chinese",
    "es": "spanish",
    "ne": "nepali",
    "sr": "serbian",
    "sd": "sindhi",
    "fa": "persian",
    "lt": "lithuanian",
    "et": "estonian",
    "in": "indonesian",  # Deprecated, use 'id' in modern standards
    "ja": "japanese",
    "tr": "turkish",
    "ar": "arabic",
    "ru": "russian",
    "ko": "korean",
    "de": "german",
    "zxx": "no linguistic content",
    "ckb": "central kurdish",
    "qam": "armenian (Twitter-specific)",
    "ml": "malayalam",
    "no": "norwegian",
    "pl": "polish",
    "lv": "latvian",
    "art": "artificial language",
    "bg": "bulgarian",
    "or": "oriya",
    "uk": "ukrainian",
    "mr": "marathi",
    "hi": "hindi",
    "te": "telugu",
    "si": "sinhala",
    "kn": "kannada",
    "gu": "gujarati"
}

MIN_APP_FREC = 5
steemer_dict = {
        "en": LancasterStemmer(),
        **{lan : SnowballStemmer(language_map[lan]) for lan in language_map.keys() if lan!="en" and language_map[lan] in SnowballStemmer.languages}
        }

stop_words_dict = {
    **{lan : set(stopwords.words(language_map[lan])) for lan in language_map.keys() if language_map[lan] in stopwords.fileids()} 
}

mentions_pattrn = re.compile(r'(@\w+|http\S+|#\w+)')
number_words_pattrn = re.compile(r'\b\w*\d\w*\b')


