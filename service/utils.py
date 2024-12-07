import re

from service.config import CONF_MAPPER
from service.models import ConfName


def matches_pattern(pattern, string):
    escaped_pattern = re.escape(pattern)
    escaped_pattern = escaped_pattern.replace(r'\*', '.*')
    regex_pattern = f'^{escaped_pattern}$'
    return bool(re.match(regex_pattern, string))


def get_configuration_name(url: str) -> ConfName:
    for key, value in CONF_MAPPER.items():
        if matches_pattern(key, url):
            return value
    return ConfName.UNKNOWN