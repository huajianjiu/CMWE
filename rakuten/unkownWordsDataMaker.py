import re, string, pickle, numpy, mojimoji, datetime, os, sys
from janome.tokenizer import Tokenizer as JanomeTokenizer
from tqdm import tqdm
import random

REVIEW_DIR = '/media/yuanzhike/D4B8C1ADB8C18E84/楽天データ/ichiba/review/'
TRAIN_SIZE_LIMIT = 80 * 10000
TUNE_SIZE_LIMIT = 10 * 10000
VALIDATION_SIZE_LIMIT = 10000
TEST_SIZE_LIMIT = 10000

# TODO: GET THE WORD VOCABULARY AND CHARACTER VOCABULARY of TRAIN SET
# TODO: TEST SET of DIFFERENT UNKOWN WORD/CHARACTER PERCENTAGE

janome_tokenizer = JanomeTokenizer()


def read_one_file(fpath):
    with open(fpath) as f:
        for line in f.readlines():
            rank = int(line.split("\t")[13])
            review_text = line.split("\t")[15]
            # print(rank, review_text)
            if rank >= 3:
                label = 1
            elif rank < 3:
                label = 0
            else:
                continue  # discard the middle ranks to avoid ambiguous texts
    return review_text, label


def parse_text(text, label, set_d):
    parse_tokens = janome_tokenizer.tokenize(text)
    for token in parse_tokens:
        word = token.surface
        if word not in set_d["word_vocab"]:
            set_d["word_vocab"].append(word)
    for character in text:
        if character not in set_d["character_vocab"]:
            set_d["character_vocab"].append(character)
    if label == 0:
        set_d["negative"].append(text)
    elif label == 1:
        set_d["positive"].append(text)
    else:
        print("Unhandled Label")
        raise
    return True


def parse_text_for_unk_w(text, label, set_d, vocabulary):
    ok_flag = False
    words = []
    characters = []
    parse_tokens = janome_tokenizer.tokenize(text)
    for token in parse_tokens:
        word = token.surface
        if word not in words:
            words.append(word)
        if word not in vocabulary:
            ok_flag = True
    for character in text:
        if character not in characters:
            characters.append(character)
    if ok_flag:
        set_d["word_vocab"] = words
        set_d["character_vocab"] = characters
        if label == 0:
            set_d["negative"].append(text)
        elif label == 1:
            set_d["positive"].append(text)
        else:
            print("Unhandled Label")
            raise
        return True
    else:
        return False


def parse_text_for_unk_c(text, label, set_d, vocabulary):
    ok_flag = False
    words = []
    characters = []
    parse_tokens = janome_tokenizer.tokenize(text)
    for token in parse_tokens:
        word = token.surface
        if word not in words:
            words.append(word)
    for character in text:
        if character not in characters:
            characters.append(character)
        if character not in vocabulary:
            ok_flag = True
    if ok_flag:
        set_d["word_vocab"] = words
        set_d["character_vocab"] = characters
        if label == 0:
            set_d["negative"].append(text)
        elif label == 1:
            set_d["positive"].append(text)
        else:
            print("Unhandled Label")
            raise
        return True
    else:
        return False


train_set = {"positive": [], "negative": [], "word_vocab": [], "character_vocab": []}
tune_set = {"positive": [], "negative": [], "word_vocab": [], "character_vocab": []}
validation_set = {"positive": [], "negative": [], "word_vocab": [], "character_vocab": []}
test_normal_set = {"positive": [], "negative": [], "word_vocab": [], "character_vocab": []}
test_unk_w_set = {"positive": [], "negative": [], "word_vocab": [], "character_vocab": []}
test_unk_c_set = {"positive": [], "negative": [], "word_vocab": [], "character_vocab": []}


file_list = sorted(os.listdir(REVIEW_DIR))
random.shuffle(file_list)
private_file_count = 0

for file_count, fname in enumerate(tqdm(file_list)):
    fpath = os.path.join(REVIEW_DIR, fname)
    # print(fpath)
    if file_count < TRAIN_SIZE_LIMIT:
        # obtain trainning_data:
        review_text, label = read_one_file(fpath)
        parse_text(review_text, label, train_set)
    elif file_count < TRAIN_SIZE_LIMIT + TUNE_SIZE_LIMIT:
        # obtian tuning data:
        review_text, label = read_one_file(fpath)
        parse_text(review_text, label, tune_set)
    elif file_count < TRAIN_SIZE_LIMIT + TUNE_SIZE_LIMIT + VALIDATION_SIZE_LIMIT:
        # obtain val data:
        review_text, label = read_one_file(fpath)
        parse_text(review_text, label, validation_set)
    elif file_count < TRAIN_SIZE_LIMIT + TUNE_SIZE_LIMIT + VALIDATION_SIZE_LIMIT + TEST_SIZE_LIMIT:
        # obtain normal test data:
        review_text, label = read_one_file(fpath)
        parse_text(review_text, label, test_normal_set)
    elif file_count < TRAIN_SIZE_LIMIT + TUNE_SIZE_LIMIT + VALIDATION_SIZE_LIMIT + 2 * TEST_SIZE_LIMIT:
        # obtain 100% unkown word test data (at least one unkown word in each sentence):
        pass
    elif file_count < TRAIN_SIZE_LIMIT + TUNE_SIZE_LIMIT + VALIDATION_SIZE_LIMIT + 2 * TEST_SIZE_LIMIT:
        # obtain 100% unkown Chinese character test data (at least one character word in each sentence):
        pass

with open("rakuten_review_split.pickle", "wb") as f:
    pickle.dump((train_set, tune_set, validation_set, test_normal_set, test_unk_w_set, test_unk_c_set), f)
