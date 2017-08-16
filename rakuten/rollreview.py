import re, string, pickle, numpy, mojimoji, datetime, os, sys

REVIEW_DIR = '/media/yuanzhike/D4B8C1ADB8C18E84/楽天データ/ichiba/review/'
SIZE_LIMIT = 100*1000
positive = []
negative = []

for fname in sorted(os.listdir(REVIEW_DIR)):
    fpath = os.path.join(REVIEW_DIR, fname)
    # print(fpath)
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


print("Size of pos and neg sets:", len(positive), len(negative))
with open("rakuten_review.pickle", "wb") as f:
    pickle.dump((positive, negative), f)
