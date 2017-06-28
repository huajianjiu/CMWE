# -*- coding: utf-8 -*-

def strip_ideographic(text):
    # Ideographic_Description_Characters = ["⿰", "⿱", "⿲", "⿳", "⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺", "⿻"]
    Ideographic_Description_Characters = "⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻"
    translator = str.maketrans("", "", Ideographic_Description_Characters)
    return text.translate(translator)


def get_all_word_bukken(filename="IDS-UCS-Basic.txt"):
    bukkens = []
    words = []
    word_bukken = {}
    count = 0
    for i, line in enumerate(open(filename, "r").readlines()):
        if line[0] != "U": # not start with U+XXXX means it is not a word
            continue
        line = line.split()
        word = line[1]
        components = line[2]
        components = strip_ideographic(components)
        bukken = []
        while ";" in components:
            bukken.append(components[:components.find(";")+1])
            components = components[components.find(";")+1:]
        while len(components)>1:
            bukken.append(components[0])
            components = components[1:]
        bukken.append(components)
        words.append(word)
        word_bukken[word] = bukken
        if len(bukken) == 1 and bukken[0] == word:
            bukkens.append(word)
    def expand_bukken(bukken):
        expanded_bukken=[]
        for b in bukken:
            if b in bukkens:
                expanded_bukken.append(b)
            else:
                if b in words:
                    expanded_bukken.append(expand_bukken(word_bukken[b]))
                else:
                    bukkens.append(b)
                    expanded_bukken.append(b)
        return expanded_bukken
    for word, bukken in word_bukken.items():
        word_bukken[word] = expand_bukken(bukken)
    return words, bukkens, word_bukken

if __name__ == "__main__":
    # print(strip_ideographic('⿱⿰&CDP-895C;&CDP-895C;一'))
    _, _, word_bukken = get_all_word_bukken("IDS-UCS-test.txt")
    for word, bukken in word_bukken.items():
        print(word + ": " + str(bukken))

