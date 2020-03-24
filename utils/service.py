from utils.constant import ktp

import re


def regex_extractor(lines, types, prefix):
    cards = dict()
    for line in lines:
        for key, value in ktp.items():
            match = re.search(value, line)
            if match:
                start = 1
                end = 0
                if key in ['TTL']:
                    start = 3
                cards[key] = word_extractor(match.group(0), start, end)

    return cards


def size_thresh(image):
    return image.shape[1] < 450


def blur_detection(score):
    thres = 600
    normal = 3000

    if score > normal:
        return 5
    elif score > thres and score < normal:
        return 11
    else:
        return 3


def image_orientation(angle):
    if angle in ['0', '180']:
        return True
    return False


def propotional(old, new):
    for i in range(2):
        if new.shape[i] < (old.shape[i] / 4):
            print('crop too small: ', new.shape[i], old.shape[i])
            return False
    return True


def scale_image(width, height):
    if width > 1000:
        return 80
    elif width > 500 and width < 1000:
        return 100
    else:
        return 120


def card_type(text):
    types = None
    prefix = 0

    # prefix identifier
    if bool(re.search(r'(Nama|Alamat)', text)):
        prefix = 1

    # card type
    if bool(re.search(r'(NI(K|X))', text)):
        types = 'KTP'
    elif bool(re.search(r'(METRO|JAYA)', text)):
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
    return re.sub(r'([^\s\w-])+', '', text)
