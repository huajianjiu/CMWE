vocabulary_path = "../HistoryData/term_vocabulary_ja.txt"

with open(vocabulary_path, "r") as f:
    f_content = f.read()
    vocabulary = [x for x in f_content.split("\n") if x != ""]


