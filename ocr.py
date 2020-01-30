from models import Ktp
from constant import *

import cv2
import json
import re
import pytesseract as ts


def read_card(encoded):
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    prep = preprocessing(image)
    data = ts.image_to_string(prep)
    card = card_classifier(data)
    return json.dumps(card.__dict__)


def preprocessing(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, threshold = cv2.threshold(
        blurred, 127, 255, cv2.THRESH_TRUNC+cv2.THRESH_OTSU)

    return threshold


def card_classifier(text):
    if 'NIK' in text:
        return extract_ktp(text)
    else:
        # TODO
        pass


def extract_ktp(text):
    text = remove_punctuation(text)
    lines = text.splitlines()

    ktp = Ktp()

    flag = None

    for line in lines:

        data, sep = get_data(line)

        if next((key for key in nik if key in line), False):
            if sep:
                ktp.nik = data
            else:
                ktp.nik = inline(data, 1)
        elif next((key for key in name if key in line), False):
            if sep:
                ktp.nama = data
            else:
                ktp.nama = inline(data, 1)
            flag = 'name'
        elif next((key for key in gender if key in line), False):
            if sep:
                ktp.jenis_kelamin = inline(data, 0, 1)
            else:
                ktp.jenis_kelamin = inline(data, 2, 3)
        elif next((key for key in birthday if key in line), False):
            if sep:
                ktp.ttl = data
            else:
                ktp.ttl = inline(data, 2)
        elif next((key for key in address if key in line), False):
            if sep:
                ktp.alamat = data
            else:
                ktp.alamat = inline(data, 1)
            flag = 'addr'
        elif next((key for key in areas if key in line), False):
            if next((key for key in neighbourhood if key in line), False):
                if sep:
                    ktp.alamat += ' RT/RW {}'.format(data)
                else:
                    ktp.alamat += ' RT/RW {}'.format(inline(data, 1))
            else:
                if sep:
                    ktp.alamat += ' {}'.format(data)
                else:
                    ktp.alamat += ' {}'.format(inline(data, 1))
        elif next((key for key in religion if key in line), False):
            ktp.agama = inline(data, 1, 2)
        elif next((key for key in status if key in line), False):
            if sep:
                ktp.status_perkawinan = inline(data, 0, -1)
            else:
                ktp.status_perkawinan = inline(data, 2)
        elif next((key for key in jobs if key in line), False):
            if sep:
                ktp.pekerjaan = data
            else:
                ktp.pekerjaan = inline(data, 1)
        elif next((key for key in citizenship if key in line), False):
            if sep:
                ktp.kewarganegaraan = data
            else:
                ktp.kewarganegaraan = inline(data, 1)
        elif next((key for key in validity if key in line), False):
            if sep:
                ktp.berlaku_hingga = data
            else:
                ktp.berlaku_hingga = inline(data, 2)
        else:
            if flag == 'name':
                ktp.nama += ' {}'.format(data)
                flag = None
            elif flag == 'addr':
                ktp.alamat += ' {}'.format(data)
                flag = None

    return ktp


def get_data(text):
    if ":" in text:
        return " ".join(text.split(':')[1:]).strip(), True
    return text, False


def inline(text, start=0, end=0):
    if end == 0:
        end = len(text.split(' '))
    return " ".join(text.split(' ')[start:end]).strip()


def remove_punctuation(text):
    punctuation = '''!"#$%&'()*+,.;<=>?@[\]^_`{|}~'''
    table = str.maketrans(
        {key: None for key in punctuation})
    return text.translate(table)
