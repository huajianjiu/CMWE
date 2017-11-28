import re, string, pickle, numpy, mojimoji, datetime, os, sys
from janome.tokenizer import Tokenizer as JanomeTokenizer

REVIEW_DIR = '/media/yuanzhike/D4B8C1ADB8C18E84/楽天データ/ichiba/review/'
TRAIN_SIZE_LIMIT = 80*10000
TUNE_SIZE_LIMIT = 10*10000
VALIDATION_SIZE_LIMIT = 10*10000
TEST_SIZE_LIMIT = 10*10000
positive = []
negative = []

#TODO: GET THE WORD VOCABULARY AND CHARACTER VOCABULARY of TRAIN SET
#TODO: TEST SET of DIFFERENT UNKOWN WORD/CHARACTER PERCENTAGE

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

for file_count, fname in enumerate(sorted(os.listdir(REVIEW_DIR))):
    fpath = os.path.join(REVIEW_DIR, fname)
    # print(fpath)
    if file_count < TRAIN_SIZE_LIMIT:
        #obtain trainning_data:
        with open(fpath) as f:
            for line in f.readlines():
                rank = int(line.split("\t")[13])
                review_text = line.split("\t")[15]
                # print(rank, review_text)
                if rank > 4 and len(positive) < SIZE_LIMIT:
                    positive.append(review_text)
                elif rank < 2 and len(negative) < SIZE_LIMIT:
                    negative.append(review_text)
                elif len(positive) >= SIZE_LIMIT or len(negative) >= SIZE_LIMIT:
                    break
                else:
                    continue  # discard the middle ranks to avoid ambiguous texts
    elif file_count < TRAIN_SIZE_LIMIT + TUNE_SIZE_LIMIT:
        #obtian tuning data:
        pass
    elif file_count < TRAIN_SIZE_LIMIT + TUNE_SIZE_LIMIT + VALIDATION_SIZE_LIMIT:
        #obtain val data:
        pass
    elif file_count < TRAIN_SIZE_LIMIT + TUNE_SIZE_LIMIT + VALIDATION_SIZE_LIMIT + TEST_SIZE_LIMIT:
        #obtain 50% unkown test data:
        pass
    elif file_count < TRAIN_SIZE_LIMIT + TUNE_SIZE_LIMIT + VALIDATION_SIZE_LIMIT + 2*TEST_SIZE_LIMIT:
        # obtain 75% unkown test data:
        pass
    elif file_count < TRAIN_SIZE_LIMIT + TUNE_SIZE_LIMIT + VALIDATION_SIZE_LIMIT + 3 * TEST_SIZE_LIMIT:
        # obtain 100% unkown test data:
        pass


print("Size of pos and neg sets:", len(positive), len(negative))
with open("rakuten_review.pickle", "wb") as f:
    pickle.dump((positive, negative), f)
