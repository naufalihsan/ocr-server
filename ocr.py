from models import Ktp
from constant import *

import cv2
import json
import re
import string
import numpy as np
import pytesseract as ts


def read_card(encoded):
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    prep = preprocessing(image)
    data = ts.image_to_string(prep)
    card = card_classifier(data)
    return json.dumps(card)


def preprocessing(image):
    resized = resize(image)
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (11, 11), 0)
    _, threshold = cv2.threshold(
        blurred, 127, 255, cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(threshold, cv2.MORPH_HITMISS, kernel)

    return morph


def resize(image):
    scale_percent = 100
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)


def card_classifier(text):
    text = remove_punctuation(text)
    lines = text.splitlines()
    cards = dict()
    for line in lines:
        print(line)
        for key, value in query.items():
            match = re.search(value, line)
            if match:
                cards[key] = word_extractor(match.group(0), 1)

    if cards.get('NIK'):
        cards['Type'] = 'KTP'
    else:
        cards['Type'] = 'Other'

    return cards


def word_extractor(text, start=0, end=0):
    splitted = text.split(' ')
    if end == 0:
        end = len(splitted)
    return " ".join(splitted[start:end]).strip()


def remove_punctuation(text):
    table = str.maketrans(
        {key: None for key in string.punctuation})
    return text.translate(table)
