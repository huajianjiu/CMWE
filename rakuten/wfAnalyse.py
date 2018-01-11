import matplotlib.pyplot as plt
import os, pickle
from tqdm import tqdm

REVIEW_DIR = '/media/yuanzhike/D4B8C1ADB8C18E84/楽天データ/ichiba/review/'

VOCABUALRY = {}


def parse_text(text):
    for character in text:
        if character not in list(VOCABUALRY.keys()):
            VOCABUALRY[character] = 1
        else:
            VOCABUALRY[character] += 1
    return True


def read_one_file(fpath):
    with open(fpath) as f:
        for line in tqdm(f.readlines()):
            review_text = line.split("\t")[15]
            parse_text(review_text)

file_list = sorted(os.listdir(REVIEW_DIR))

for fname in tqdm(file_list):
    fpath = os.path.join(REVIEW_DIR, fname)
    read_one_file(fpath)

VOCABUALRY = sorted(VOCABUALRY.items(), key=lambda d: d[1], reverse=True)
with open("frequency_vocab.pkl", "wb") as f:
    pickle.dump(VOCABUALRY, f)

