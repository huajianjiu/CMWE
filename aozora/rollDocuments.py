import pandas
import pickle
import numpy
from six import string_types

MAX_DOCUMENTS = 500
MAX_SENTENCE_LENGTH = 100
MAX_SENTENCE_NUM = 1000

document_list = pandas.read_csv("aozora_word_list_utf8.csv")
label_names = ["NDC 3", "NDC 7", "NDC 9"]
class_1 = []
class_2 = []
class_3 = []
rows, columns = document_list.shape
for i in range(rows):
    category = document_list.get_value(i, "分類番号")
    if isinstance(category, string_types):
        category = category[:5]
    if category in label_names:
        d_file_name = document_list.get_value(i, "file")
        if category == label_names[0] and len(class_1) < MAX_DOCUMENTS:
            class_1.append(d_file_name)
        if category == label_names[1] and len(class_2) < MAX_DOCUMENTS:
            class_2.append(d_file_name)
        if category == label_names[2] and len(class_3) < MAX_DOCUMENTS:
            class_3.append(d_file_name)
print(len(class_1), len(class_2), len(class_3))
del document_list


def getI(w, d):
    try:
        return d[w]
    except KeyError:
        return None

data = {}
for n, class_n in enumerate([class_1, class_2, class_3]):
    if class_n:
        class_reverse = {}
        for i, w in enumerate(class_n):
            class_reverse[w] = i
        class_sequences = []
        cur_document = ""
        word_sequence = []
        flat_sequences = []
        sample_sequences = []
        with open("aozora-newnew.csv") as f:
            for line in f.readlines():
                line = line.split(",")
                document_file = line[0]
                if document_file != cur_document:
                    cur_document = document_file
                    if word_sequence:
                        class_sequences.append(word_sequence)
                        word_sequence = []
                if getI(cur_document, class_reverse) is not None:
                    word = line[3]
                    word_sequence.append(word)
                    flat_sequences.append(word)
        print(len(class_sequences))
        # data[label_names[n]] = class_sequences[:200]
        num_words = len(flat_sequences)
        num_split_pos = (num_words-1) // MAX_SENTENCE_LENGTH
        random_split_poses = numpy.random.randint(low=0, high=num_split_pos, size=MAX_SENTENCE_NUM)
        for split_pose in random_split_poses:
            sample_sequences.append(flat_sequences[split_pose:split_pose+MAX_SENTENCE_LENGTH])
        data[label_names[n]] = sample_sequences
    else:
        continue


with open("aozora_3.pickle", "wb") as f:
    pickle.dump(data, f)