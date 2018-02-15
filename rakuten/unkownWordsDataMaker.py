# generate balanced p/n review dataset of Training, Tuning, Validation, Test, UNKTest

import pickle, os
from janome.tokenizer import Tokenizer as JanomeTokenizer
from tqdm import tqdm
import random

REVIEW_DIR = '/media/yuanzhike/D4B8C1ADB8C18E84/楽天データ/ichiba/review/'
TRAIN_SIZE_LIMIT = 10000 # half, number of positive/negative
TUNE_SIZE_LIMIT = 1000
VALIDATION_SIZE_LIMIT = 1000
TEST_SIZE_LIMIT = 1000
# 2017 12 4: very little negative samples -> reduce the data number to get balanced dataset

# TODO: GET THE WORD VOCABULARY AND CHARACTER VOCABULARY of TRAIN SET
# TODO: TEST SET of DIFFERENT UNKOWN WORD/CHARACTER PERCENTAGE

janome_tokenizer = JanomeTokenizer()


def read_one_file(fpath, type, set_d, limit):
    p_count = 0
    n_count = 0
    p_limit = limit - len(set_d["positive"])
    n_limit = limit - len(set_d["negative"])
    with open(fpath) as f:
        for line in f.readlines():
            if (p_count + n_count) % 1000 == 0:
                print (p_count, n_count)
            if p_count >= p_limit and n_count >= n_limit:
                break
            rank = int(line.split("\t")[13])
            review_text = line.split("\t")[15]
            # print("Rank: ", rank, " Text: ", review_text[:20], "...")
            # print(rank, review_text)
            if rank > 4 and p_count < limit:
                label = 1
            elif rank < 2 and n_count < limit:
                label = 0
            else:
                continue
            if type == "unk_w":
                result = parse_text_for_unk_w(review_text, label, set_d)
            elif type == "unk_c":
                result = parse_text_for_unk_c(review_text, label, set_d)
            else:
                result = parse_text(review_text, label, set_d)
            if result and label == 1:
                p_count += 1
            elif result and label == 0:
                n_count += 1
    return p_count + n_count


def parse_text(text, label, set_d):
    try:
        parse_tokens = janome_tokenizer.tokenize(text)
    except:
        return False
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


def parse_text_for_unk_w(text, label, set_d, vocabulary=None):
    if not vocabulary:
        vocabulary = train_set["word_vocab"]
    ok_flag = False
    words = []
    characters = []
    try:
        parse_tokens = janome_tokenizer.tokenize(text)
    except:
        return False
    for token in parse_tokens:
        word = token.surface
        if (word not in words) and (word not in set_d["word_vocab"]):
            words.append(word)
        if word not in vocabulary:
            ok_flag = True
    for character in text:
        if (character not in characters) and (character not in set_d["character_vocab"]):
            characters.append(character)
    if ok_flag:
        set_d["word_vocab"] += words
        set_d["character_vocab"] += characters
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


def parse_text_for_unk_c(text, label, set_d, vocabulary=None):
    if not vocabulary:
        vocabulary = train_set["character_vocab"]
    ok_flag = False
    words = []
    characters = []
    try:
        parse_tokens = janome_tokenizer.tokenize(text)
    except:
        return False
    for token in parse_tokens:
        word = token.surface
        if (word not in words) and (word not in set_d["word_vocab"]):
            words.append(word)
    for character in text:
        if (character not in characters) and (character not in set_d["character_vocab"]):
            characters.append(character)
        if character not in vocabulary:
            ok_flag = True
    if ok_flag:
        set_d["word_vocab"] += words
        set_d["character_vocab"] += characters
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

review_count = 0

for fname in tqdm(file_list):
    fpath = os.path.join(REVIEW_DIR, fname)
    # print(fpath)
    print("Count: ", review_count)
    if review_count < 2 * TRAIN_SIZE_LIMIT:
        print("Training set")
        # obtain trainning_data:
        review_count += read_one_file(fpath, "train", train_set, TRAIN_SIZE_LIMIT)
    elif review_count < 2 * (TRAIN_SIZE_LIMIT + TUNE_SIZE_LIMIT):
        print("Tuning set")
        # obtian tuning data:
        review_count += read_one_file(fpath, "tune", tune_set, TUNE_SIZE_LIMIT)
    elif review_count < 2 * (TRAIN_SIZE_LIMIT + TUNE_SIZE_LIMIT + VALIDATION_SIZE_LIMIT):
        print("Validation set")
        # obtain val data:
        review_count += read_one_file(fpath, "validation", validation_set, VALIDATION_SIZE_LIMIT)
    elif review_count < 2 * (TRAIN_SIZE_LIMIT + TUNE_SIZE_LIMIT + VALIDATION_SIZE_LIMIT + TEST_SIZE_LIMIT):
        print("Normal Test set")
        # obtain normal test data:
        review_count += read_one_file(fpath, "test", test_normal_set, TEST_SIZE_LIMIT)
    elif review_count < 2 * (TRAIN_SIZE_LIMIT + TUNE_SIZE_LIMIT + VALIDATION_SIZE_LIMIT + 2 * TEST_SIZE_LIMIT):
        print("UNK W Test set")
        # obtain 100% unkown word test data (at least one unkown word in each sentence):
        review_count += read_one_file(fpath, "unk_w", test_unk_w_set, TEST_SIZE_LIMIT)
    elif review_count < 2 * (TRAIN_SIZE_LIMIT + TUNE_SIZE_LIMIT + VALIDATION_SIZE_LIMIT + 3 * TEST_SIZE_LIMIT):
        print("UNK C Test set")
        # obtain 100% unkown Chinese character test data (at least one character word in each sentence):
        review_count += read_one_file(fpath, "unk_c", test_unk_c_set, TEST_SIZE_LIMIT)
    else:
        break

with open("rakuten_review_split_only1and5.pickle", "wb") as f:
    pickle.dump((train_set, tune_set, validation_set, test_normal_set, test_unk_w_set, test_unk_c_set), f)
