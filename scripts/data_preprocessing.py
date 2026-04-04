"""
Preprocessing bilingual dictionary 
for exploring word embeddings in context
of Ukrainian-English translation
"""
import csv
import re

from translitua import (
    translit,
    UkrainianKMU,
    UkrainianSimple,
    UkrainianWWS,
    UkrainianBritish,
    UkrainianBGN,
    UkrainianISO9,
    UkrainianPassport2007,
    UkrainianNational1996,
)

TRANSLIT_SYSTEMS = [
    UkrainianKMU,
    UkrainianSimple,
    UkrainianWWS,
    UkrainianBritish,
    UkrainianBGN,
    UkrainianISO9,
    UkrainianPassport2007,
    UkrainianNational1996,
]

def clean_word(word: str)->str:
    """
    Function cleans the word, if it starts from some wrong punctuation
    (unneccessary symbols at the beginning or the ending of the word)

    :param word: str, word to clean
    :return: str, cleaned word
    """
    return re.sub(r"^[^a-zA-Zа-щьюяєіїґ'-]+|[^a-zA-Zа-щьюяєіїґ'-]+$", "", word)

def check_word_language(word: str, language: str)->bool:
    """
    Function checks if the word is language correct 
    according to Regex.

    :param word: str, word to check
    :param language: str, language of the word
    :return: bool, result of checking
    """
    match language:
        case "ua":
            return bool(re.fullmatch(r"[а-щьюяєіїґ]+(?:[-'][а-щьюяєіїґ]+)*", word))
        case "en":
            return bool(re.fullmatch(r"[a-z]+(?:[-'][a-z]+)*", word))
    return False

def convert_transliteracy(word: str) -> list:
    """
    Function finds all transliteracy variant of the Ukrainian word.

    :param word: str, word to transliterate
    :return: list, transliteration results
    """
    return set(translit(word, system) for system in TRANSLIT_SYSTEMS)

def clean_transliteracy(dictionary: dict, ua_word: str, transliterations: list)-> dict:
    """
    Function removes transliterated variants of Ukrainian words from the dictionary.

    :param dictionary: dict, dictionary with Ukrainian-English word pairs
    :param ua_word: str, Ukrainian word to clean
    :param transliterations: list, list of transliteration variants
    :return: dict, cleaned dictionary
    """
    all_words = dictionary[ua_word]
    words_to_be_removed = []

    if len(all_words) == 1:
        if all(all_words[0] == word for word in transliterations):
            return dictionary
        elif any(all_words[0] == word for word in transliterations):
            dictionary.pop(ua_word)
    elif len(all_words) > 1:
        for word in all_words:
            if word in transliterations:
                words_to_be_removed.append(word)

        for word in words_to_be_removed:
            dictionary[ua_word].remove(word)

        if not dictionary[ua_word]:
            dictionary.pop(ua_word)

    return dictionary


def write_res_csv(data_dictionary: dict, path: str)-> None:
    """
    Function writes the cleaned words into
    csv file

    :param data_dictionary: dict, dictionary with Ukrainian-English words
    :param path: str, path to save csv file
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(["ukrainian", "english"])

        for ua, eng_collection in data_dictionary.items():
            for eng in eng_collection:
                writer.writerow([ua, eng])

def data_preprocessing(original_dictionary: str="../data/original/uk-en-full.txt",
                       cleaned_dictionary: str="../data/usage/uk-en-full.csv") -> dict:
    """
    Function cleans the original English-Ukrainian dictionary
    and forms new dictionary with word pairs only.

    :param original_dictionary: str, path to a file with original dictionary
    :param cleaned_dictionary: str, path to a file with cleaned dictionary
    :return: dict, processed dictionary of Ukrainian-English pairs
    """
    result_dictionary = {}
    with open(original_dictionary, "r", encoding="utf-8") as f:
        data = f.readlines()

    for line in data:
        line = line.strip("\n")
        if not line:
            continue

        ukrainian_word, english_word = line.split()

        ukrainian_word = ukrainian_word.lower()
        english_word = english_word.lower()

        ukrainian_word = clean_word(ukrainian_word)
        english_word = clean_word(english_word)

        if check_word_language(ukrainian_word, "ua") and check_word_language(english_word, "en"):
            result_dictionary.setdefault(ukrainian_word, [])
            if english_word not in result_dictionary[ukrainian_word]:
                result_dictionary[ukrainian_word].append(english_word)

    for ukrainian_word in list(result_dictionary.keys()):
        transliterated_ukrainian_word = convert_transliteracy(ukrainian_word)
        result_dictionary = clean_transliteracy(result_dictionary, ukrainian_word, 
                                                transliterated_ukrainian_word)

    write_res_csv(result_dictionary, cleaned_dictionary)
    return result_dictionary

data_preprocessing()
