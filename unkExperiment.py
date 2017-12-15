import re, string, pickle, numpy, pandas, mojimoji, random, os, jieba, sys
import tensorflow as tf
from keras import optimizers
from keras.models import Model
from keras.layers import Embedding, Input, AveragePooling1D, MaxPooling1D, Conv1D, concatenate, TimeDistributed, \
    Bidirectional, LSTM, Dense, Flatten, GRU, Lambda
from keras.legacy.layers import Highway
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard
from keras.utils.np_utils import to_categorical
from attention import AttentionWithContext
from getShapeCode import get_all_word_bukken, get_all_character
from janome.tokenizer import Tokenizer as JanomeTokenizer
from keras import backend as K
from tqdm import tqdm
from plot_results import plot_results, save_curve_data
from dataReader import prepare_char, prepare_word, shuffle_kv

from ShapeEmbedding import build_fasttext, build_hatt, build_sentence_rnn, build_word_feature_char, \
    build_word_feature_shape, text_to_char_index, get_vocab, _make_kana_convertor, train_model, test_model

# MAX_SENTENCE_LENGTH = 739  # large number as 739 makes cudnn die
MAX_SENTENCE_LENGTH = 500
MAX_WORD_LENGTH = 4
COMP_WIDTH = 3
CHAR_EMB_DIM = 15
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
BATCH_SIZE = 100
WORD_DIM = 600
MAX_RUN = 1
VERBOSE = 0
EPOCHS = 50


def shuffle_data_one_set(data_shape, data_char, data_word, labels):
    # split data into training and validation
    indices = numpy.arange(data_char.shape[0])
    numpy.random.shuffle(indices)
    data_shape = data_shape[indices]
    data_word = data_word[indices]
    data_char = data_char[indices]
    labels = labels[indices]
    return data_shape, data_char, data_word, labels


def prepare_set_j(s, full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin,
                  preprocessed_char_number, word_vocab, char_vocab, janome_tokenizer):
    n_hira_punc_number_latin = len(hira_punc_number_latin) + 2
    positive = s["positive"]
    negative = s["negative"]
    labels = [1] * len(positive) + [0] * len(negative)
    data_size = len(positive) + len(negative)
    data_shape = numpy.zeros((data_size, MAX_SENTENCE_LENGTH, COMP_WIDTH * MAX_WORD_LENGTH), dtype=numpy.int32)
    data_char = numpy.zeros((data_size, MAX_SENTENCE_LENGTH, MAX_WORD_LENGTH), dtype=numpy.int32)
    data_word = numpy.zeros((data_size, MAX_SENTENCE_LENGTH), dtype=numpy.int32)
    for i, text in enumerate(tqdm(positive + negative)):
        # 日语分词
        janome = True
        parse_tokens = janome_tokenizer.tokenize(text)
        for j, mrph in enumerate(parse_tokens):
            if j + 1 > MAX_SENTENCE_LENGTH:
                break
            if janome:
                word = mrph.surface
            else:
                word = mrph.midasi
            if word not in word_vocab:
                word_vocab.append(word)
                word_index = len(word_vocab) - 1
            else:
                word_index = word_vocab.index(word)
            data_word[i, j] = word_index
            # Single char gram level
            # Convert digital number and latin to hangaku
            word = mojimoji.zen_to_han(word, kana=False)
            # Convert kana to zengaku
            word = mojimoji.han_to_zen(word, digit=False, ascii=False)
            # Convert kata to hira - should not. cuz katakana is used mainly for name entity
            # _, katakana2hiragana, _ = _make_kana_convertor()
            # word = katakana2hiragana(word)
            # word = word.translate(additional_translate)
            # Lowercase
            word = word.lower()
            for l, char_g in enumerate(word):
                if char_g not in char_vocab:
                    char_vocab.append(char_g)
                    char_g_index = len(char_vocab) - 1
                else:
                    char_g_index = char_vocab.index(char_g)
                if l < MAX_WORD_LENGTH:
                    data_char[i, j, l] = char_g_index
                if char_g not in full_vocab:
                    full_vocab.append(char_g)
            # char shape level
            char_index = text_to_char_index(full_vocab=full_vocab, real_vocab_number=real_vocab_number,
                                            chara_bukken_revised=chara_bukken_revised,
                                            addition_translate=additional_translate,
                                            sentence_text=word, preprocessed_char_number=preprocessed_char_number,
                                            skip_unknown=False, shuffle=None)
            if len(char_index) < COMP_WIDTH * MAX_WORD_LENGTH:
                char_index = char_index + [0] * (COMP_WIDTH * MAX_WORD_LENGTH - len(char_index))  # Padding
            elif len(char_index) > COMP_WIDTH * MAX_WORD_LENGTH:
                char_index = char_index[:COMP_WIDTH * MAX_WORD_LENGTH]
            for k, comp in enumerate(char_index):
                data_shape[i, j, k] = comp
    labels = to_categorical(numpy.asarray(labels))
    print('Label Shape:', labels.shape)
    x_shape, x_char, x_word, y = shuffle_data_one_set(data_shape, data_char, data_word, labels)
    return x_shape, x_char, x_word, y

def print_vocab_size(full_vocab, word_vocab, char_vocab):
    print("full_vocab\tword_vocab\tchar_vocab")
    print(len(full_vocab), "\t", len(word_vocab), "\t", len(char_vocab))

def unk_experiment_j():
    train_set, tune_set, validation_set, test_normal_set, test_unk_w_set, test_unk_c_set \
        = pickle.load(open("rakuten/rakuten_review_split.pickle", "rb"))

    janome_tokenizer = JanomeTokenizer()
    full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin = get_vocab()
    preprocessed_char_number = len(full_vocab)
    word_vocab = ["</s>"]
    char_vocab = ["</s>"] + get_all_character()

    print_vocab_size(full_vocab, word_vocab, char_vocab)
    x_s_train, x_c_train, x_w_train, y_train = prepare_set_j(train_set, full_vocab, real_vocab_number,
                                                             chara_bukken_revised, additional_translate,
                                                             hira_punc_number_latin, preprocessed_char_number,
                                                             word_vocab, char_vocab, janome_tokenizer)
    print_vocab_size(full_vocab, word_vocab, char_vocab)
    x_s_validation, x_c_validation, x_w_validation, y_validation = prepare_set_j(validation_set, full_vocab,
                                                                                 real_vocab_number,
                                                                                 chara_bukken_revised,
                                                                                 additional_translate,
                                                                                 hira_punc_number_latin,
                                                                                 preprocessed_char_number,
                                                                                 word_vocab, char_vocab,
                                                                                 janome_tokenizer)
    print_vocab_size(full_vocab, word_vocab, char_vocab)
    x_s_test_normal, x_c_test_normal, x_w_test_normal, y_test_normal = prepare_set_j(test_normal_set, full_vocab,
                                                                                     real_vocab_number,
                                                                                     chara_bukken_revised,
                                                                                     additional_translate,
                                                                                     hira_punc_number_latin,
                                                                                     preprocessed_char_number,
                                                                                     word_vocab, char_vocab,
                                                                                     janome_tokenizer)
    print_vocab_size(full_vocab, word_vocab, char_vocab)
    x_s_test_unk_w, x_c_test_unk_w, x_w_test_unk_w, y_test_unk_w = prepare_set_j(test_unk_w_set, full_vocab,
                                                                                     real_vocab_number,
                                                                                     chara_bukken_revised,
                                                                                     additional_translate,
                                                                                     hira_punc_number_latin,
                                                                                     preprocessed_char_number,
                                                                                     word_vocab, char_vocab,
                                                                                     janome_tokenizer)
    print_vocab_size(full_vocab, word_vocab, char_vocab)
    x_s_test_unk_c, x_c_test_unk_c, x_w_test_unk_c, y_test_unk_c = prepare_set_j(test_unk_c_set, full_vocab,
                                                                                     real_vocab_number,
                                                                                     chara_bukken_revised,
                                                                                     additional_translate,
                                                                                     hira_punc_number_latin,
                                                                                     preprocessed_char_number,
                                                                                     word_vocab, char_vocab,
                                                                                     janome_tokenizer)
    word_vocab_size = len(word_vocab)
    char_vocab_size = len(char_vocab)
    num_class = 2
    data_set_name = "Rakuten_UNK"

    model_name = "Radical-CNN-RNN"
    print("======MODEL: ", model_name, "======")
    model = build_sentence_rnn(real_vocab_number=real_vocab_number, classes=num_class,
                               char_shape=True, word=False, char=False,
                               cnn_encoder=True, highway=None, nohighway="linear",
                               attention=True, shape_filter=True, char_filter=True)
    print("Train")
    train_model(model, x_s_train, y_train, x_s_validation, y_validation, model_name, path="unk_exp/")
    print("Test-Normal")
    test_model(model, model_name, x_s_test_normal, y_test_normal, path="unk_exp/")
    print("Test-UNK-WORDS")
    test_model(model, model_name, x_s_test_unk_w, y_test_unk_w, path="unk_exp/")
    print("Test-UNK-CHAR")
    test_model(model, model_name, x_s_test_unk_c, y_test_unk_c, path="unk_exp/")

    model_name = "CHAR-CNN-RNN"
    print("======MODEL: ", model_name, "======")
    model = build_sentence_rnn(real_vocab_number=real_vocab_number, classes=num_class,
                               char_shape=False, word=False, char=True,
                               cnn_encoder=True, highway=None, nohighway="linear",
                               attention=True, shape_filter=True, char_filter=True)
    print("Train")
    train_model(model, x_c_train, y_train, x_c_validation, y_validation, model_name, path="unk_exp/")
    print("Test-Normal")
    test_model(model, model_name, x_c_test_normal, y_test_normal, path="unk_exp/")
    print("Test-UNK-WORDS")
    test_model(model, model_name, x_c_test_unk_w, y_test_unk_w, path="unk_exp/")
    print("Test-UNK-CHAR")
    test_model(model, model_name, x_c_test_unk_c, y_test_unk_c, path="unk_exp/")

    model_name = "WORD-RNN"
    print("======MODEL: ", model_name, "======")
    model = build_sentence_rnn(real_vocab_number=real_vocab_number, classes=num_class,
                               char_shape=False, word=True, char=False,
                               cnn_encoder=True, highway=None, nohighway="linear",
                               attention=True, shape_filter=True, char_filter=True)
    print("Train")
    train_model(model, x_w_train, y_train, x_w_validation, y_validation, model_name, path="unk_exp/")
    print("Test-Normal")
    test_model(model, model_name, x_w_test_normal, y_test_normal, path="unk_exp/")
    print("Test-UNK-WORDS")
    test_model(model, model_name, x_w_test_unk_w, y_test_unk_w, path="unk_exp/")
    print("Test-UNK-CHAR")
    test_model(model, model_name, x_w_test_unk_c, y_test_unk_c, path="unk_exp/")

    model_name = "WORD-HATT"
    print("======MODEL: ", model_name, "======")
    model = build_hatt(word_vocab_size, 2)
    print("Train")
    train_model(model, x_w_train, y_train, x_w_validation, y_validation, model_name, path="unk_exp/")
    print("Test-Normal")
    test_model(model, model_name, x_w_test_normal, y_test_normal, path="unk_exp/")
    print("Test-UNK-WORDS")
    test_model(model, model_name, x_w_test_unk_w, y_test_unk_w, path="unk_exp/")
    print("Test-UNK-CHAR")
    test_model(model, model_name, x_w_test_unk_c, y_test_unk_c, path="unk_exp/")

    model_name = "WORD-FASTTEXT"
    print("======MODEL: ", model_name, "======")
    model = build_fasttext(word_vocab_size, 2)
    print("Train")
    train_model(model, x_w_train, y_train, x_w_validation, y_validation, model_name, path="unk_exp/")
    print("Test-Normal")
    test_model(model, model_name, x_w_test_normal, y_test_normal, path="unk_exp/")
    print("Test-UNK-WORDS")
    test_model(model, model_name, x_w_test_unk_w, y_test_unk_w, path="unk_exp/")
    print("Test-UNK-CHAR")
    test_model(model, model_name, x_w_test_unk_c, y_test_unk_c, path="unk_exp/")


if __name__ == "__main__":
    unk_experiment_j()
