from joblib import load
from utils.models import Ktp
from utils.pipeline import Pipeline

import cv2
import json
import re
import string
import numpy as np
import pytesseract as ts


BLURRED_THRES = 600


def read_card(encoded):
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    crop = region_of_interest(image)
    prep = preprocessing(crop)
    data = ts.image_to_string(prep)
    card = card_classifier(data)
    return json.dumps(card)


def region_of_interest(image):
    prep = precropping(image)
    roi = find_countour(prep, image)
    return roi


def preprocessing(image):
    print('after crop size: ', image.shape)
    resized = resize(image)
    print('after resized: ', resized.shape)
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    is_blurred = variance_of_laplacian(grayscale)
    print('blur', is_blurred)
    if is_blurred < BLURRED_THRES:
        blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
    else:
        blurred = cv2.GaussianBlur(grayscale, (9, 9), 0)
    _, threshold = cv2.threshold(
        blurred, 127, 255, cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    morph = cv2.morphologyEx(threshold, cv2.MORPH_HITMISS, kernel)
    # bnw = cv2.adaptiveThreshold(morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return morph


def precropping(image):
    print('before crop size: ', image.shape)
    dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    grayscale = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, threshold = cv2.threshold(
        blurred, 127, 255, cv2.THRESH_BINARY)

    return threshold


def find_countour(prep, original):
    contours, _ = cv2.findContours(
        prep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    old = original.copy()
    flag = False

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            if ar > 1 and ar < 2:
                new = old[y:y + h, x:x + w]
                if propotional(old, new):
                    roi = new
                    flag = True

    if not flag:
        roi = old

    return roi


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def propotional(old, new):
    for i in range(2):
        if new.shape[i] < (old.shape[i] / 4):
            print('crop too small: ', new.shape[i], old.shape[i])
            return False
    return True


def resize(image):
    scale_percent = scale_image(image.shape[1])
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)


def scale_image(width):
    if width < 700:
        return 120
    else:
        return 100


def card_classifier(text):
    print(text)
    classifier = dict()

    text = remove_punctuation(text).lower()
    types, prefix = card_type(text)
    lines = text.splitlines()
    clean = [word_extractor(l, prefix) for l in lines]
    clf = Pipeline(clean)
    preds = clf.classifier(model='gbc')

    classifier['type'] = types
    classifier['data'] = filter_preds(preds, types)

    return classifier


def filter_preds(preds, types):
    filtered = dict()
    lines = 0
    for pred in preds:
        entity, category = pred
        if not category in filtered:
            filtered[category] = entity
        elif types == 'KTP' and category == 'name':
            if lines == 1:
                filtered['name'] = entity
            lines += 1

    return filtered


def card_type(text):
    types = None
    prefix = 0

    # prefix identifier
    if bool(re.search('(nama|alamat)', text)):
        prefix = 1

    # card type
    if bool(re.search('(nik)', text)):
        types = 'KTP'
    elif bool(re.search('(metro|jaya)', text)):
        types = 'SIM'
    else:
        types = 'Other'

    return types, prefix


def word_extractor(text, start=0, end=0):
    splitted = text.split(' ')
    if end == 0:
        end = len(splitted)
    return " ".join(splitted[start:end]).strip()


def remove_punctuation(text):
    return re.sub(r'([^\s\w]|_)+', ' ', text)
