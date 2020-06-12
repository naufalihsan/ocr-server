from joblib import load
from utils.pipeline import Pipeline
from utils.service import *
from PIL import Image

import cv2
import json
import re
import tempfile
import string
import numpy as np
import pprint
import pytesseract as ts


def read_card(encoded, orientation=0, algorithm='gbc', parser='regex'):

    err = False
    msg = None

    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    print('before=>', image.shape)

    obj_card = detect_object(image)
    obj_card = deskew_object(obj_card)
    prep = image_processing(obj_card)

    if size_thresh(prep):
        err = True
        msg = {'error': f'gambar {image.shape} terlalu kecil'}

    if not err and orientation:
        osd = ts.image_to_osd(prep)

        angle = re.search(r'(?<=Rotate: )\d+', osd).group(0)

        if not image_orientation(angle):
            err = True
            msg = {'error': f'posisi gambar {angle} derajat'}

    if not err:
        data = ts.image_to_string(prep)
        result = card_classifier(data, algorithm, parser)
        return json.dumps(result)

    return json.dumps(msg)


def detect_object(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return detect_contour(mask, image)


def detect_contour(mask, original):
    old = original.copy()

    flag = False
    roi = None

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            if ar > 1.5 and ar < 2:
                new = old[y:y + h, x:x + w]
                if propotional(old, new):
                    roi = new
                    flag = True

    if not flag:
        roi = old
        print(type(roi))

    return roi


def detect_skewness(image):
    _, width = get_image_size(image)
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            100, minLineLength=width / 4.0, maxLineGap=20)

    skewness = 0.0
    angle = 0.0

    try:
        nlines = lines.size

        for l in range(len(lines)):
            for x1, y1, x2, y2 in lines[l]:
                angle += np.arctan2(y2 - y1, x2 - x1)

        skewness = (angle/nlines)

    except Exception as e:
        print(e)

    print('skew=>', skewness)

    return skewness


def deskew_object(image):
    height, width = get_image_size(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bitwise = cv2.bitwise_not(gray)
    angle = detect_skewness(bitwise)
    non_zero_pixels = cv2.findNonZero(bitwise)
    center, _, _ = cv2.minAreaRect(non_zero_pixels)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, rotation_matrix, (height, width), flags=cv2.INTER_CUBIC)
    return cv2.getRectSubPix(rotated, (height, width), center)


def image_processing(image):
    tempfile = rescale_image(image)
    rescaled = cv2.imread(tempfile, cv2.IMREAD_UNCHANGED)
    print("after=>", rescaled.shape)
    grayscale = cv2.cvtColor(rescaled, cv2.COLOR_BGR2GRAY)
    blur_score = variance_of_laplacian(grayscale)
    size = blur_detection(blur_score)
    blurred = cv2.GaussianBlur(grayscale, (size, size), 0)
    _, threshold = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    morph = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

    return morph


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def rescale_image(image):
    im_pil = Image.fromarray(image)
    width, height = im_pil.size
    size = get_new_size(width, height)

    try:
        resized = im_pil.resize(size, Image.BICUBIC)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_filename = temp_file.name
        resized.save(temp_filename, dpi=(300, 300))
    except Exception as e:
        print(e)

    return temp_filename


def card_classifier(text, algorithm, parser):
    classifier = dict()
    text = remove_punctuation(text)
    print(text)
    types, prefix = card_type(text)
    lines = text.splitlines()

    if parser == 'regex':
        preds = regex_extractor(lines, types, prefix)
    else:
        clean = [word_extractor(line, prefix) for line in lines]
        print(clean)
        clf = Pipeline(clean)
        preds = clf.classifier(model=algorithm)

    classifier['type'] = types
    classifier['data'] = preds

    pprint.pprint(classifier)

    return classifier


def precropping(image):
    print('before crop size =>', image.shape)
    dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    grayscale = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, threshold = cv2.threshold(
        blurred, 127, 255, cv2.THRESH_BINARY)

    return threshold
