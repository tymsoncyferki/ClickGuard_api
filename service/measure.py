import string 
import numpy as np
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
import textstat

import math
import re

from textblob import TextBlob

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

STOP_WORDS = stopwords.words("english")

BAIT_PHRASES = [
 'obsessed',
 'celebrity',
 'anyone',
 'know',
 'af',
 'movies',
 'amazing',
 'totally',
 'video',
 'things',
 'remember',
 'videos',
 'quiz',
 'guess',
 'characters',
 'hilarious',
 'tweets',
 'favorite',
 'products',
 'ways',
 'pictures',
 'photos',
 'based',
 'understand',
 'perfectly',
 'actually',
 'makeup',
 'character',
 'reasons',
 'laugh',
 'nude',
 'nudes',
 'naked',
 'and it',
 'are you',
 'based on',
 'can you',
 'do you',
 'here what',
 'how to',
 'if you',
 'is the',
 'make you',
 'need to',
 'of the',
 'on your',
 'people who',
 'that are',
 'that will',
 'the best',
 'the most',
 'the world',
 'things you',
 'this is',
 'ways to',
 'we know',
 'will make',
 'you will',
 'you need',
 'you are']

""" helper functions """

def remove_punctuation(text):
    """ removes punctuation """
    return text.translate(str.maketrans('', '', string.punctuation))

def preprocess_text(text):
    """ preprocess data for embedding model training """
    if isinstance(text, list):
        text = ' '.join(text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]
    return tokens

def common_words_ratio(text):
    """ percentage of stop words in text """
    words = word_tokenize(remove_punctuation(text.lower()))
    common_words = [word for word in words if word in STOP_WORDS]
    return len(common_words) / len(words)

def capital_letters_ratio(text):
    """ calculates capital letters ratio """
    text = remove_punctuation(text)
    return sum([char.isupper() for char in list(text)]) / len(text)

def clickbait_punctuation_count(text):
    """ counts clickbait related punctuation, only left brackets as otherwise they would be double calculated """
    punctuation = '!"#(?'
    return sum([1 for x in text if x in punctuation])

def numbers_count(text):
    """ calculates the count of numbers (not digits)"""
    text = remove_punctuation(text)
    return sum([x.isnumeric() for x in word_tokenize(text)])

def pronouns_2nd_person_count(text):
    """ 2nd person pronouns usage """
    second_person_pronouns = ["you", "your", "yours", "yourself", "yourselves"]
    return sum([1 for x in word_tokenize(text.lower()) if x in second_person_pronouns])

def superlatives_ratio(text):
    """ percentage of adjectives and adverbs which are in superlative form """
    tagged_words = pos_tag(word_tokenize(text.lower()))
    adj_adv_count = 0
    superlative_count = 0
    for _, tag in tagged_words:
        if tag in ('JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'):
            adj_adv_count += 1
            if tag in ('JJS', 'RBS'):  # if superlative
                superlative_count += 1
    if adj_adv_count == 0:
        return 0
    return superlative_count / adj_adv_count

def speculatives_count(text):
    """ usage of speculative forms """
    speculative_words = {
        "might", "may", "could", "can", "perhaps", "possibly", "probably", "maybe",
        "seems", "appears", "suggests", "indicates", "likely", "unlikely", "assume"
    }
    return sum([1 for x in word_tokenize(text.lower()) if x in speculative_words])

def baiting_words_count(text):
    """ number of baiting words used """
    return sum([1 for x in BAIT_PHRASES if x in text.lower()])

def polarity_score(text):
    """ polarity score """
    return TextBlob(text).sentiment.polarity

def subjectivity_score(text):
    """ subjectivity score """
    return TextBlob(text).sentiment.subjectivity

def flesch_reading_ease_score(text):
    """ difficulty of the text, if its easy to read or not, the higher the easier
    
    Formula:
    206.835 - 1.015 * (total words / total sentences) - 84.6 (total syllables / total words)

    Scores:
    -90.00  	Very easy to read. Easily understood by an average 11-year-old student.
    90.0-80.0	Easy to read. Conversational English for consumers.
    80.0-70.0	Fairly easy to read.
    70.0-60.0	Plain English. Easily understood by 13- to 15-year-old students.
    60.0-50.0	Fairly difficult to read.
    50.0-30.0	Difficult to read.
    30.0-10.0	Very difficult to read. Best understood by university graduates.
    10.0-   	Extremely difficult to read. Best understood by university graduates.
    source: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
    """
    return textstat.flesch_reading_ease(text)


def cut_value(value):
    if value > 1:
        return 1
    elif value < 0:
        return 0
    return value

def baitness_measure(text, debug=False):

    # eye catchingness
    punct_count = clickbait_punctuation_count(text)
    capitals_ratio = capital_letters_ratio(text) * 3
    number_count = numbers_count(text)

    eye_catch_list = [punct_count, capitals_ratio, number_count]
    eye_catch = cut_value(np.mean(eye_catch_list))
    if debug:
        print(eye_catch_list)
    
    # content curiosity
    pronouns_2_count = pronouns_2nd_person_count(text)
    super_count = superlatives_ratio(text) * 2
    spec_count = speculatives_count(text)
    bait_words = baiting_words_count(text)

    curiosity_list = [pronouns_2_count, super_count, spec_count, bait_words]
    curiosity = np.sqrt(cut_value(np.mean(curiosity_list)))
    if debug:
        print(curiosity_list)

    # sentiment - measures high polarity and high subjectivity
    sentiment = math.sqrt(abs(polarity_score(text)) * subjectivity_score(text))

    # difficulty of reading - reading ease score and common words ratio
    fres = cut_value(flesch_reading_ease_score(text) / 100)
    cw_ratio = cut_value(common_words_ratio(text) * 1.5)
    ease_of_text = np.mean([fres, cw_ratio])

    metric_list = [eye_catch, sentiment, ease_of_text, curiosity]
    measure = np.mean(metric_list)
    if debug:
        print(metric_list)
    
    return measure

def explain_baitness(text, probability):
    # eye catchingness
    punct_count = clickbait_punctuation_count(text)
    capitals_ratio = capital_letters_ratio(text) * 3
    number_count = numbers_count(text)
    
    # content curiosity
    pronouns_2_count = pronouns_2nd_person_count(text)
    super_count = superlatives_ratio(text) * 2
    spec_count = speculatives_count(text)
    bait_words = baiting_words_count(text)
    
    # sentiment - measures high polarity and high subjectivity
    polarity = abs(polarity_score(text))
    subjectivity = subjectivity_score(text)

    # difficulty of reading - reading ease score and common words ratio
    fres = cut_value(flesch_reading_ease_score(text) / 100)
    cw_ratio = cut_value(common_words_ratio(text) * 1.5)

    primary_metrics = {
        "2nd person pronouns": pronouns_2_count,
        "capital words": capitals_ratio, 
        "numbers": number_count,
        "punctuation": punct_count,
        "bait words": bait_words,
        "superlatives": super_count,
        "speculatives": spec_count,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "common words": cw_ratio / 1.5,
        "reading ease": fres
    }

    sorted_metrics = sorted(primary_metrics.items(), key=lambda x: x[1], reverse=True)

    explanations = {
        "2nd person pronouns": "uses second person pronouns",
        "capital words": "uses many capital letters",
        "numbers": "uses numbers to draw attention",
        "punctuation": "uses excessive punctuation",
        "bait words": "includes clickbait-specific words",
        "superlatives": "uses superlative words",
        "speculatives": "uses speculative words",
        "polarity": "has a strong emotional tone",
        "subjectivity": "is highly subjective",
        "common words": "uses common words frequently",
        "reading ease": "is very easy to read"
    }

    print(np.sqrt(baitness_measure(text)))
    if probability > 0.6:
        top_contributors = sorted_metrics[:2]
        explanation = ", ".join([explanations[metric] for metric, _ in top_contributors])
    elif probability > 0.4:
        explanation = explanations[sorted_metrics[0][0]]
    else:
        explanation = "no grounds to classify as a clickbait"
    
    return explanation
