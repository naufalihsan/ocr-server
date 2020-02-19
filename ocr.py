from models import Ktp
from constant import *

import cv2
import json
import re
import string
import numpy as np
import pytesseract as ts


BLURRED_THRES = 1000


def read_card(encoded):
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    crop = region_of_interest(image)
    prep = preprocessing(crop)
    data = ts.image_to_string(prep)
    card = card_classifier(data)
    return json.dumps(card)


def region_of_interest(image):
    prep = precropping(image)
    roi, opt = find_countour(prep, image)
    if opt:
        print('masuk')
        return roi
    return image


def preprocessing(image):
    print(image.shape)
    resized = resize(image)
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    is_blurred = variance_of_laplacian(grayscale)
    # print(is_blurred)
    if is_blurred < BLURRED_THRES:
        blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
    else:
        blurred = cv2.GaussianBlur(grayscale, (13, 13), 0)
    _, threshold = cv2.threshold(
        blurred, 127, 255, cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    morph = cv2.morphologyEx(threshold, cv2.MORPH_HITMISS, kernel)
    # bnw = cv2.adaptiveThreshold(morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return morph


def precropping(image):
    dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    grayscale = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (11, 11), 0)
    _, threshold = cv2.threshold(
        blurred, 127, 255, cv2.THRESH_TOZERO)

    return threshold


def find_countour(prep, original):
    contours, _ = cv2.findContours(
        prep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    copy = original.copy()
    optimized = False

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            if ar > 1 and ar < 2:
                roi = copy[y:y + h, x:x + w]
                optimized = True

    return roi, optimized


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def resize(image):
    # print(image.shape)
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
    text = remove_punctuation(text)
    print(text)
    lines = text.splitlines()
    cards = dict()
    for line in lines:
        if bool(re.search('(NIK)', text)):
            cards['Type'] = 'KTP'
            for key, value in ktp.items():
                match = re.search(value, line)
                if match:
                    cards[key] = word_extractor(match.group(0), 1)
        elif bool(re.search('(METRO|JAYA)', text)):
            cards['Type'] = 'SIM'
        else:
            cards['Type'] = 'Other'

    return cards


def word_extractor(text, start=0, end=0):
    splitted = text.split(' ')
    if end == 0:
        end = len(splitted)
    return " ".join(splitted[start:end]).strip()


def remove_punctuation(text):
    return re.sub(r'([^\s\w]|_)+', '', text)
