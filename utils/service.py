from utils.constant import ktp

import re

def regex_extractor(lines, types, prefix):
    cards = dict()
    for line in lines:
        for key, value in ktp.items():
            match = re.search(value, line)
            if match:
                cards[key] = word_extractor(match.group(0), 1)

    return cards

def image_orientation(angle):
    if angle in ['0','180']:
        return True
    return False

# def filter_preds(preds, types):
#     filtered = dict()
#     lines = 0
#     for pred in preds:
#         entity, category = pred
#         if not category in filtered:
#             filtered[category] = entity
#         elif types == 'KTP' and category == 'name':
#             if lines == 1:
#                 filtered['name'] = entity
#             lines += 1

#     return filtered

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
    if bool(re.search(r'(NIK)', text)):
        types = 'KTP'
    elif bool(re.search(r'(Metro|Jaya)', text)):
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
    return re.sub(r'([^\s\w]|_)+', '', text)