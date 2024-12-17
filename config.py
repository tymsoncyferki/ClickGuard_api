import os
import pickle
import logging

from dotenv import load_dotenv
from nltk.corpus import stopwords
import nltk 

from dtos import ConfName

load_dotenv()

# config
class Config:
    OPEN_API_KEY = os.getenv("OPEN_API_KEY")
    MODEL_PATH = os.path.join(os.getcwd(), "models/XGB_1000_dimensions_and_informativeness_measures.pkl")
    NLTK_DATA_DIR = "nltk_data/"

# pre-click config mapper
CONF_MAPPER = {
    '*google.com*': ConfName.GOOGLE,
    '*thesun.co.uk*': ConfName.THESUN
}

# model loader
with open(Config.MODEL_PATH, 'rb') as rf_file:
    MODEL = pickle.load(rf_file)

# nltk data load options
nltk.data.path.append(Config.NLTK_DATA_DIR)
if os.getenv("DOWNLOAD_NLTK"):
    nltk.download('stopwords', download_dir=Config.NLTK_DATA_DIR)
    nltk.download('punkt_tab', download_dir=Config.NLTK_DATA_DIR)
    nltk.download('averaged_perceptron_tagger_eng', download_dir=Config.NLTK_DATA_DIR)

STOP_WORDS = stopwords.words("english")

# logger initialization
logging.basicConfig()
logger = logging.getLogger("clickguard")
logger.setLevel(logging.DEBUG)