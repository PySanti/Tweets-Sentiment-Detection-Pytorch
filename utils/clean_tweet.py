import emoji
import string
from .constants import *


def clean_tweet(text, lan) -> str:
    """
        Elimina emojis, @s y urls de lost tweets.
    """
    text = text.strip()
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = emoji.replace_emoji(text, replace="")
    text = mentions_pattrn.sub('', text)
    text = number_words_pattrn.sub('', text)
    try:
        if (type(lan) == str) and ('http' not in lan) and (language_map[lan] in SnowballStemmer.languages) :
            text = " ".join([steemer_dict[lan].stem(w).lower() for w in text.split(" ") if w not in stop_words_dict[lan]] )
    except Exception as e:
        print(e)
    finally:
        return text
