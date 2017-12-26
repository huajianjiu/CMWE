import pickle
import numpy
from tqdm import tqdm
from ShapeEmbedding import get_vocab, text_to_char_index, COMP_WIDTH, MAX_WORD_LENGTH
from keras.preprocessing.sequence import pad_sequences, skipgrams
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Embedding, Input, AveragePooling1D, MaxPooling1D, Conv1D, concatenate, TimeDistributed, \
    Bidirectional, LSTM, Dense, Flatten, GRU, Lambda
from keras.legacy.layers import Highway

term_vocabulary_path = "../HistoryData/term_vocabulary_ja.txt"

full_vocab, real_vocab_number, chara_bukken_revised, addtional_translate, hira_punc_number_latin = get_vocab()
preprocessed_char_number = len(full_vocab)

with open(term_vocabulary_path, "r") as f_vocab:
    f_content = f_vocab.read()
    term_vocabulary = [x for x in f_content.split("\n") if x != ""]


def generate_pair(filepath=None):
    pairs = []
    context_window = 16
    if filepath == None:
        filepath = "../HistoryData/ja.txt"
    with open(filepath, "r") as f_corpus:
        _text = f_corpus.read()
    text = _text.split(" ")
    sliced_text = [text[i:i + context_window] for i in range(0, len(text), context_window)]
    for slice_t in sliced_text[:5]:
        for i in range(len(slice_t)):
            # target: the index of target word in term_vocabulary
            # context: the index sequence of the radical-level component of the other words
            target = slice_t[i]
            if target not in term_vocabulary:
                term_vocabulary.append(target)
            pair = {"target": term_vocabulary.index(target), "context": []}
            context = []
            for word in slice_t[0:i] + slice_t[i + 1:]:
                text_int_list = text_to_char_index(full_vocab=full_vocab, real_vocab_number=real_vocab_number,
                                                   chara_bukken_revised=chara_bukken_revised,
                                                   addition_translate=addtional_translate,
                                                   sentence_text=word,
                                                   preprocessed_char_number=preprocessed_char_number)
                context += text_int_list
            pair["context"] = context
            pairs.append(pair)
    with open("../HistoryData/sliced_pairs_radical.pickle", "wb") as f_save_slice:
        pickle.dump(pairs, f_save_slice)
    with open("../HistoryData/term_vocabulary_after_train.pickle", "wb") as f_save_tv:
        pickle.dump(term_vocabulary, f_save_tv)


def generate_skipgram_pair(filepath=None):
    context_window = 16
    if filepath == None:
        filepath = "../HistoryData/ja.txt"
    with open(filepath, "r") as f_corpus:
        _text = f_corpus.read()
    text_indices = []
    for word in tqdm(_text.split(" ")):
        if word not in term_vocabulary:
            term_vocabulary.append(word)
        text_indices.append(term_vocabulary.index(word))
    sampling_table = numpy.zeros(shape=(len(term_vocabulary),))
    for index in tqdm(text_indices):
        sampling_table[index] = sampling_table[index] + 1
    couples, labels = skipgrams(text_indices, len(term_vocabulary), window_size=context_window,
                                negative_samples=5., categorical=False, sampling_table=sampling_table)
    x1 = numpy.zeros((len(couples),), dtype=numpy.int32)
    x2 = numpy.zeros((len(couples), COMP_WIDTH * MAX_WORD_LENGTH))
    for i, couple in enumerate(couples):
        x1[i] = couple[0]
        context_word = term_vocabulary[couple[1]]
        if len(context_word) > MAX_WORD_LENGTH:
            context_word = context_word[:MAX_WORD_LENGTH]
        context_int_list = text_to_char_index(full_vocab=full_vocab, real_vocab_number=real_vocab_number,
                                              chara_bukken_revised=chara_bukken_revised,
                                              addition_translate=addtional_translate,
                                              sentence_text=context_word,
                                              preprocessed_char_number=preprocessed_char_number)
        for j in context_int_list:
            x2[i, j] = context_int_list[j]
    y = to_categorical(numpy.asarray(labels))
    with open("../HistoryData/sliced_pairs_skipgram_x1x2y.pickle", "wb") as f_save_slice:
        pickle.dump((x1, x2, y), f_save_slice)
    with open("../HistoryData/term_vocabulary_after_train.pickle", "wb") as f_save_tv:
        pickle.dump(term_vocabulary, f_save_tv)

def build_radical_cnn(radical_vocab_size, radical_emb_dim, radical_width, input_length, output_classes):
    init_width = 0.5 / radical_emb_dim
    init_weight = numpy.random.uniform(low=-init_width, high=init_width, size=(radical_vocab_size, radical_emb_dim))
    init_weight[0] = 0  # maybe the padding should not be zero
    # print(init_weight)
    # first layer embeds
    #  every components
    word_input = Input(shape=(radical_width * input_length,))
    char_embedding = \
        Embedding(input_dim=radical_vocab_size, output_dim=radical_emb_dim, weights=[init_weight], trainable=True)(
            word_input)
    filter_sizes = [50, 100, 150]
    feature_s1 = Conv1D(filters=filter_sizes[0], kernel_size=1, activation='relu')(
        char_embedding)
    feature_s1 = MaxPooling1D(pool_size=input_length * radical_width)(feature_s1)
    feature_s2 = Conv1D(filters=filter_sizes[1], kernel_size=2, activation='relu')(
        char_embedding)
    feature_s2 = MaxPooling1D(pool_size=input_length * radical_width - 1)(feature_s2)
    feature_s3 = Conv1D(filters=filter_sizes[2], kernel_size=3, activation='relu')(
        char_embedding)
    feature_s3 = MaxPooling1D(pool_size=input_length * radical_width - 2)(feature_s3)
    feature1 = Conv1D(filters=filter_sizes[0], kernel_size=1 * radical_width, strides=radical_width,
                      activation='relu')(
        char_embedding)
    feature1 = MaxPooling1D(pool_size=input_length - 1 + 1)(feature1)
    feature2 = Conv1D(filters=filter_sizes[1], kernel_size=2 * radical_width, strides=radical_width,
                      activation='relu')(
        char_embedding)
    feature2 = MaxPooling1D(pool_size=input_length - 2 + 1)(feature2)
    feature3 = Conv1D(filters=filter_sizes[2], kernel_size=3 * radical_width, strides=radical_width,
                      activation='relu')(
        char_embedding)
    feature3 = MaxPooling1D(pool_size=input_length - 3 + 1)(feature3)
    feature = concatenate([feature_s1, feature_s2, feature_s3, feature1, feature2, feature3])
    feature = Flatten()(feature)
    feature = Highway(activation='relu')(feature)
    l_dens = Dense(output_classes * 5, activation="linear")(feature)
    preds = Dense(output_classes, activation='softmax')(l_dens)
    word_feature_encoder = Model(word_input, feature)
    return word_feature_encoder


if __name__ == "__main__":
    # generate_pair()
    generate_skipgram_pair()
    # with open("../HistoryData/sliced_pairs_radical.pickle", "rb") as f_save_slice:
    #     pairs = pickle.load(f_save_slice)
    # with open("../HistoryData/term_vocabulary_after_train.pickle", "rb") as f_save_tv:
    #     term_vocabulary = pickle.load(f_save_tv)
    # x_train = pad_sequences([p["context"] for p in pairs])
    # y_train = to_categorical(numpy.asarry([p["target"] for p in pairs]))
