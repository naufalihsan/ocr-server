import re

from utils.constant import ktp


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
    if score > 600 and score < 3000:
        return 11
    else:
        return 5


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
    if width > 1200:
        return 0.9
    elif width > 600 and width < 1200:
        return 1
    else:
        return 1.1


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
    splitted = text.lower().replace('-', ' ').split(' ')
    splitted = [w for w in splitted if len(
        w) > 2 or (w.isnumeric() and int(w) > 9)]

    if end == 0:
        end = len(splitted)

    return " ".join(splitted[start:end]).strip()


def remove_punctuation(text):
    return re.sub(r'([^\s\w-])+', '', text)


def convert_to_dict(data):
    card = dict()
    for k, v in data:
        if k not in card:
            if k == 'nik' or 'nik' in card:
                card[k] = v
    return card


def get_image_size(image):
    return image.shape[1], image.shape[0]


def get_image_factor(width):
    # print(width)
    if width < 800:
        return 2.1
    elif width >= 800 and width < 1200:
        return 1.5
    else:
        return 1


def get_new_size(width, height):
    factor = get_image_factor(width)
    return int(factor * width), int(factor * height)
