import warnings
import string 

import numpy as np
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
import textstat
from lexical_diversity import lex_div

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

# def get_embedding(sentence):
#     """ gets embeddings """
#     word_embeddings = [MODEL.wv[word] for word in sentence if word in MODEL.wv]
#     if not word_embeddings: 
#         return np.zeros(MODEL.vector_size)
#     sentence_embedding = np.mean(word_embeddings, axis=0)
#     return sentence_embedding

def words_count(text):
    """ calculates number of words """
    return len(text.split())

def characters_count(text):
    """ calculates number of characters including whitespaces """
    return len(text.strip())

def avg_word_length(text):
    """ calculates average word length """
    return float(np.mean([len(word) for word in word_tokenize(remove_punctuation(text))]))

def common_words_ratio(text):
    """ percentage of stop words in text """
    words = word_tokenize(remove_punctuation(text.lower()))
    common_words = [word for word in words if word in STOP_WORDS]
    return len(common_words) / len(words)

def capital_letters_ratio(text):
    """ calculates capital letters ratio """
    text = remove_punctuation(text)
    return sum([char.isupper() for char in list(text)]) / len(text)

def capital_words_count(text):
    """ calculates capital words """
    return sum([1 for word in word_tokenize(text) if word.isupper() and len(word) > 1])

def punctuation_ratio(text):
    """ calculates punctuation ratio """
    return sum([1 for x in text if x in string.punctuation]) / len(text)

def clickbait_punctuation_count(text):
    """ counts clickbait related punctuation, only left brackets as otherwise they would be double calculated """
    punctuation = '!"#(?'
    return sum([1 for x in text if x in punctuation])

def nonclickbait_punctuation_count(text):
    """ counts non-clickbait related punctuation """
    punctuation = "$%&,.;:-/"
    return sum([1 for x in text if x in punctuation])

def numbers_count(text):
    """ calculates the count of numbers (not digits)"""
    text = remove_punctuation(text)
    return sum([x.isnumeric() for x in word_tokenize(text)])

def pronouns_count(text):
    """ pronouns usage """
    tagged = pos_tag(word_tokenize(text.lower()))
    return sum([1 for x in tagged if x[1] in ('PRP', 'PRP$', 'WP', 'WP$')])

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

# def similarity_score(title, body):
#     """ custom word2vec model embeddings and cosine similarity """
#     title_embedding = get_embedding(word_tokenize(title.lower()))
#     body_embedding = get_embedding(preprocess_text(word_tokenize(body.lower())))
#     return (1 - cosine(title_embedding, body_embedding))

def polarity_score(text):
    """ polarity score """
    return TextBlob(text).sentiment.polarity

def subjectivity_score(text):
    """ subjectivity score """
    return TextBlob(text).sentiment.subjectivity

def type_token_ratio(text):
    """ lexical richness, the higher the higher diversity """
    words = word_tokenize(remove_punctuation(text.lower()))
    unique_words = set(words)
    return len(unique_words) / len(words)

def corrected_type_token_ratio(text):
    """ lexical richness, the higher the higher diversity """
    words = word_tokenize(remove_punctuation(text.lower()))
    unique_words = set(words)
    return len(unique_words) / np.sqrt(2 * len(words))

def maas_index(text):
    """ lexical diversity index, the lower the higher diversity """
    words = word_tokenize(remove_punctuation(text.lower()))
    n = len(words)
    if n == 1:
        return 0
    t = len(set(words))
    return (math.log(n) - math.log(t)) / (math.log(n) ** 2)

def hdd_metric(text):
    """ hypergeometric distribution D, lexical diversity, the higher the higher diversity """
    words = word_tokenize(remove_punctuation(text.lower()))
    if len(words) < 42:
        message = f"The text is {len(words)} words long. HD-D metric is not defined for texts with less than 42 tokens."
        warnings.warn(message, UserWarning)
    return lex_div.hdd(words)

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

def flesch_kincaid_grade_level(text):
    """ difficulty of the text, US grade level needed to understand the text quite easily, the higher the harder

    Formula:
    0.39 * (total words / total sentences) + 11.8(total syllables / total words) - 15.59

    can also mean the number of years of education generally required to understand this text, relevant when the formula results in a number greater than 10
    source: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
    """
    return textstat.flesch_kincaid_grade(text)

def automated_readability_index(text):
    """ difficulty of the text, US grade level necessary to comprehend the text, the higher the harder

    Formula:
    4.71 * (characters / words) + 0.5 (words / sentences) - 21.43

    Scores:
    score | age | grade level
    1	5-6	    Kindergarten
    2	6-7	    First Grade
    3	7-8	    Second Grade
    4	8-9	    Third Grade
    5	9-10	Fourth Grade
    6	10-11	Fifth Grade
    7	11-12	Sixth Grade
    8	12-13	Seventh Grade
    9	13-14	Eighth Grade
    10	14-15	Ninth Grade
    11	15-16	Tenth Grade
    12	16-17	Eleventh Grade
    13	17-18	Twelfth Grade
    14	18-22	College student
    source: https://en.wikipedia.org/wiki/Automated_readability_index
    """
    return textstat.automated_readability_index(text)


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

metrics_functions = {
    "n_words": words_count,
    "cw_percentage": common_words_ratio,
    "capitals_ratio": capital_letters_ratio,
    "capitals_count":capital_words_count,
    "bait_punct": clickbait_punctuation_count,
    "nonbait_punct": nonclickbait_punctuation_count,
    "numbers": numbers_count,
    "2nd_pronouns": pronouns_2nd_person_count,
    "superlatives": superlatives_ratio,
    "speculatives": speculatives_count,
    "bait_words": baiting_words_count,
    "polarity": polarity_score,
    "subjectivity": subjectivity_score,
    "fres": flesch_reading_ease_score,
}


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
        "punctuation": punct_count,
        "capital words": capitals_ratio, 
        "numbers": number_count,
        "2nd person pronouns": pronouns_2_count,
        "superlatives": super_count,
        "speculatives": spec_count,
        "bait words": bait_words,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "reading ease": fres,
        "common words": cw_ratio
    }

    sorted_metrics = sorted(primary_metrics.items(), key=lambda x: x[1], reverse=True)

    explanations = {
        "punctuation": "uses excessive punctuation",
        "capital words": "uses capital words",
        "numbers": "uses numbers to draw attention",
        "2nd person pronouns": "uses 2nd person pronouns",
        "superlatives": "uses superlative words",
        "speculatives": "uses speculative words",
        "bait words": "includes clickbait-specific words",
        "polarity": "has a strong emotional tone",
        "subjectivity": "is highly subjective",
        "reading ease": "is very easy to read",
        "common words": "uses common words frequently",
    }

    if probability > 0.6:
        top_contributors = sorted_metrics[:2]
        explanation = ", ".join([explanations[metric] for metric, _ in top_contributors])
    elif probability > 0.4:
        explanation = explanations[sorted_metrics[0][0]]
    else:
        explanation = "no grounds to classify as a clickbait"
    
    return explanation
