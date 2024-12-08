from .dtos import ConfName

class Config:

    OPEN_API_KEY = ""

CONF_MAPPER = {
    '*google.com*': ConfName.GOOGLE,
    '*thesun.co.uk*': ConfName.THESUN
}