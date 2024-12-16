from dtos import ConfName
from dotenv import load_dotenv
import os
import pickle

load_dotenv()

class Config:
    OPEN_API_KEY = os.getenv("OPEN_API_KEY")
    MODEL_PATH = os.path.join(os.getcwd(), "models/XGB_1000_dimensions_and_informativeness_measures.pkl")

CONF_MAPPER = {
    '*google.com*': ConfName.GOOGLE,
    '*thesun.co.uk*': ConfName.THESUN
}

with open(Config.MODEL_PATH, 'rb') as rf_file:
    MODEL = pickle.load(rf_file)