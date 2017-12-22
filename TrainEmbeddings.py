vocabulary_path = "../HistoryData/term_vocabulary_ja.txt"

with open(vocabulary_path, "r") as f_vocab:
    f_content = f_vocab.read()
    vocabulary = [x for x in f_content.split("\n") if x != ""]


def generate_pair(filepath=None):
    context_window = 8
    if filepath == None:
        filepath = "../HistoryData/ja.txt"
    with open(filepath, "r") as f_corpus:
        text = f_corpus.read()
    sliced_text = [text[i + context_window] for i in range(0, len(text), context_window)]

