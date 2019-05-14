import matplotlib.pyplot as plt
from matplotlib import font_manager
import os, pickle, numpy
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

def count_corpus():
    file_list = sorted(os.listdir(REVIEW_DIR))

    for fname in tqdm(file_list):
        fpath = os.path.join(REVIEW_DIR, fname)
        read_one_file(fpath)

    VOCABUALRY = sorted(VOCABUALRY.items(), key=lambda d: d[1], reverse=True)
    with open("frequency_vocab.pkl", "wb") as f:
        pickle.dump(VOCABUALRY, f)


def draw_histogram():
    with open("frequency_vocab.pkl", "rb") as f:
        v = pickle.load(f)
    names, freqs = zip(*v)
    print(names[0])
    simulate = []
    # simulate the dist and print the hist
    for i, freq in enumerate(freqs[:1000]):
        simulate += [i] * (freq//10000)
    prop = font_manager.FontProperties(fname="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")
    plt.hist(simulate, 40)
    plt.xlabel("Frequent Words ←--→ Infrequent Words")
    plt.ylabel("Frequency")
    x = [i*100 for i in range(10)]
    plt.xticks(x,
               [names[i]+"..." for i in x ],
               fontproperties=prop)
    plt.savefig("char_freq.ps", bbox_inches="tight", transparent=False)

draw_histogram()