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
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from matplotlib import pyplot as plt

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
VERBOSE = 1
EPOCHS = 30


def load_data(set='015'):
    if set=='015':
        train_set, tune_set, validation_set, test_normal_set, test_unk_w_set, test_unk_c_set\
            = pickle.load(open("rakuten/rakuten_review_split_only1and5.pickle", "rb"))
        test_unk_c_set["positive"] = test_unk_c_set["positive"][:1000]
        return train_set, tune_set, validation_set, test_normal_set, test_unk_w_set, test_unk_c_set
    elif set=='012345':
        train_set, tune_set, validation_set, test_normal_set, test_unk_w_set, test_unk_c_set \
            = pickle.load(open("rakuten/rakuten_review_split.pickle", "rb"))
        return train_set, tune_set, validation_set, test_normal_set, test_unk_w_set, test_unk_c_set
    else:
        print("illegal set name")
        exit(-1)


def plot_result(history, dirname):
    plt.clf()
    plt.figure(figsize=(4, 3))
    plt.ylim(0.0, 0.5)
    print(dir(history))
    print(dir(history.validation_data))
    print(history.history.keys())
    print(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['Test_N_loss'])
    plt.plot(history.history['Test_UNKW_loss'])
    plt.plot(history.history['Test_UNKC_loss'])
    # plt.title('model loss')
    plt.ylabel('Cross Entropy Error')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validate', 'normal test', 'unknown words', 'unknown characters'], loc='upper left')
    # plt.show()
    plt.savefig('plots/' + dirname + ".png", bbox_inches='tight')


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


def unk_exp_preproces_j():
    train_set, tune_set, validation_set, test_normal_set, test_unk_w_set, test_unk_c_set \
        = load_data()

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
    with open("unk_exp/rakuten_processed_review_split_ongly15.pickle", "wb") as f:
        pickle.dump((full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin,
                     preprocessed_char_number, word_vocab, char_vocab,
                     x_s_train, x_c_train, x_w_train, y_train,
                     x_s_validation, x_c_validation, x_w_validation, y_validation,
                     x_s_test_normal, x_c_test_normal, x_w_test_normal, y_test_normal,
                     x_s_test_unk_w, x_c_test_unk_w, x_w_test_unk_w, y_test_unk_w,
                     x_s_test_unk_c, x_c_test_unk_c, x_w_test_unk_c, y_test_unk_c), f)


def unk_experiment_j_p1():
    with open("unk_exp/rakuten_processed_review_split_ongly15.pickle", "rb") as f:
        full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin, \
        preprocessed_char_number, word_vocab, char_vocab, \
        x_s_train, x_c_train, x_w_train, y_train, \
        x_s_validation, x_c_validation, x_w_validation, y_validation, \
        x_s_test_normal, x_c_test_normal, x_w_test_normal, y_test_normal, \
        x_s_test_unk_w, x_c_test_unk_w, x_w_test_unk_w, y_test_unk_w, \
        x_s_test_unk_c, x_c_test_unk_c, x_w_test_unk_c, y_test_unk_c = pickle.load(f)
    word_vocab_size = len(word_vocab)
    char_vocab_size = len(char_vocab)
    num_class = 2
    data_set_name = "Rakuten_UNK"

    model_name = "Radical-CNN-RNN HARC"
    print("======MODEL: ", model_name, "======")
    model = build_sentence_rnn(real_vocab_number=real_vocab_number, char_vocab_size=char_vocab_size,
                               word_vocab_size=word_vocab_size, classes=num_class,
                               char_shape=True, word=False, char=False,
                               cnn_encoder=True, highway="relu", nohighway="linear",
                               attention=True, shape_filter=True, char_filter=True)
    print("Train")
    train_model(model, x_s_train, y_train, x_s_validation, y_validation, model_name, path="unk_exp/")
    print("Test-Normal")
    test_model(model, model_name, x_s_test_normal, y_test_normal, path="unk_exp/")
    print("Test-UNK-WORDS")
    test_model(model, model_name, x_s_test_unk_w, y_test_unk_w, path="unk_exp/")
    print("Test-UNK-CHAR")
    test_model(model, model_name, x_s_test_unk_c, y_test_unk_c, path="unk_exp/")

    model_name = "Radical-CNN-RNN HAR"
    print("======MODEL: ", model_name, "======")
    model = build_sentence_rnn(real_vocab_number=real_vocab_number, char_vocab_size=char_vocab_size,
                               word_vocab_size=word_vocab_size, classes=num_class,
                               char_shape=True, word=False, char=False,
                               cnn_encoder=True, highway="relu", nohighway="linear",
                               attention=True, shape_filter=True, char_filter=False)
    print("Train")
    train_model(model, x_s_train, y_train, x_s_validation, y_validation, model_name, path="unk_exp/")
    print("Test-Normal")
    test_model(model, model_name, x_s_test_normal, y_test_normal, path="unk_exp/")
    print("Test-UNK-WORDS")
    test_model(model, model_name, x_s_test_unk_w, y_test_unk_w, path="unk_exp/")
    print("Test-UNK-CHAR")
    test_model(model, model_name, x_s_test_unk_c, y_test_unk_c, path="unk_exp/")

    model_name = "Radical-CNN-RNN HAC"
    print("======MODEL: ", model_name, "======")
    model = build_sentence_rnn(real_vocab_number=real_vocab_number, char_vocab_size=char_vocab_size,
                               word_vocab_size=word_vocab_size, classes=num_class,
                               char_shape=True, word=False, char=False,
                               cnn_encoder=True, highway="relu", nohighway="linear",
                               attention=True, shape_filter=False, char_filter=True)
    print("Train")
    train_model(model, x_s_train, y_train, x_s_validation, y_validation, model_name, path="unk_exp/")
    print("Test-Normal")
    test_model(model, model_name, x_s_test_normal, y_test_normal, path="unk_exp/")
    print("Test-UNK-WORDS")
    test_model(model, model_name, x_s_test_unk_w, y_test_unk_w, path="unk_exp/")
    print("Test-UNK-CHAR")
    test_model(model, model_name, x_s_test_unk_c, y_test_unk_c, path="unk_exp/")

    # model_name = "Radical-CNN-RNN AR"
    # print("======MODEL: ", model_name, "======")
    # model = build_sentence_rnn(real_vocab_number=real_vocab_number, char_vocab_size=char_vocab_size,
    #                            word_vocab_size=word_vocab_size, classes=num_class,
    #                            char_shape=True, word=False, char=False,
    #                            cnn_encoder=True, highway=None, nohighway="linear",
    #                            attention=True, shape_filter=True, char_filter=False)
    # print("Train")
    # train_model(model, x_s_train, y_train, x_s_validation, y_validation, model_name, path="unk_exp/")
    # print("Test-Normal")
    # test_model(model, model_name, x_s_test_normal, y_test_normal, path="unk_exp/")
    # print("Test-UNK-WORDS")
    # test_model(model, model_name, x_s_test_unk_w, y_test_unk_w, path="unk_exp/")
    # print("Test-UNK-CHAR")
    # test_model(model, model_name, x_s_test_unk_c, y_test_unk_c, path="unk_exp/")
    #
    # model_name = "Radical-CNN-RNN AC"
    # print("======MODEL: ", model_name, "======")
    # model = build_sentence_rnn(real_vocab_number=real_vocab_number, char_vocab_size=char_vocab_size,
    #                            word_vocab_size=word_vocab_size, classes=num_class,
    #                            char_shape=True, word=False, char=False,
    #                            cnn_encoder=True, highway=None, nohighway="linear",
    #                            attention=True, shape_filter=False, char_filter=True)
    # print("Train")
    # train_model(model, x_s_train, y_train, x_s_validation, y_validation, model_name, path="unk_exp/")
    # print("Test-Normal")
    # test_model(model, model_name, x_s_test_normal, y_test_normal, path="unk_exp/")
    # print("Test-UNK-WORDS")
    # test_model(model, model_name, x_s_test_unk_w, y_test_unk_w, path="unk_exp/")
    # print("Test-UNK-CHAR")
    # test_model(model, model_name, x_s_test_unk_c, y_test_unk_c, path="unk_exp/")
    # #
    # model_name = "Radical-CNN-RNN ARC"
    # print("======MODEL: ", model_name, "======")
    # model = build_sentence_rnn(real_vocab_number=real_vocab_number, char_vocab_size=char_vocab_size,
    #                            word_vocab_size=word_vocab_size, classes=num_class,
    #                            char_shape=True, word=False, char=False,
    #                            cnn_encoder=True, highway=None, nohighway="linear",
    #                            attention=True, shape_filter=True, char_filter=True)
    # print("Train")
    # train_model(model, x_s_train, y_train, x_s_validation, y_validation, model_name, path="unk_exp/")
    # print("Test-Normal")
    # test_model(model, model_name, x_s_test_normal, y_test_normal, path="unk_exp/")
    # print("Test-UNK-WORDS")
    # test_model(model, model_name, x_s_test_unk_w, y_test_unk_w, path="unk_exp/")
    # print("Test-UNK-CHAR")
    # test_model(model, model_name, x_s_test_unk_c, y_test_unk_c, path="unk_exp/")
    #
    # model_name = "Radical-CNN-RNN HRC"
    # print("======MODEL: ", model_name, "======")
    # model = build_sentence_rnn(real_vocab_number=real_vocab_number, char_vocab_size=char_vocab_size,
    #                            word_vocab_size=word_vocab_size, classes=num_class,
    #                            char_shape=True, word=False, char=False,
    #                            cnn_encoder=True, highway="relu", nohighway="linear",
    #                            attention=False, shape_filter=True, char_filter=True)
    # print("Train")
    # train_model(model, x_s_train, y_train, x_s_validation, y_validation, model_name, path="unk_exp/")
    # print("Test-Normal")
    # test_model(model, model_name, x_s_test_normal, y_test_normal, path="unk_exp/")
    # print("Test-UNK-WORDS")
    # test_model(model, model_name, x_s_test_unk_w, y_test_unk_w, path="unk_exp/")
    # print("Test-UNK-CHAR")
    # test_model(model, model_name, x_s_test_unk_c, y_test_unk_c, path="unk_exp/")


def unk_experiment_j_p2():
    with open("unk_exp/rakuten_processed_review_split_ongly15.pickle", "rb") as f:
        full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin, \
        preprocessed_char_number, word_vocab, char_vocab, \
        x_s_train, x_c_train, x_w_train, y_train, \
        x_s_validation, x_c_validation, x_w_validation, y_validation, \
        x_s_test_normal, x_c_test_normal, x_w_test_normal, y_test_normal, \
        x_s_test_unk_w, x_c_test_unk_w, x_w_test_unk_w, y_test_unk_w, \
        x_s_test_unk_c, x_c_test_unk_c, x_w_test_unk_c, y_test_unk_c = pickle.load(f)
    word_vocab_size = len(word_vocab)
    char_vocab_size = len(char_vocab)
    num_class = 2
    data_set_name = "Rakuten_UNK"

    model_name = "Radical-CNN-RNN HR"
    print("======MODEL: ", model_name, "======")
    model = build_sentence_rnn(real_vocab_number=real_vocab_number, char_vocab_size=char_vocab_size,
                               word_vocab_size=word_vocab_size, classes=num_class,
                               char_shape=True, word=False, char=False,
                               cnn_encoder=True, highway="relu", nohighway="linear",
                               attention=False, shape_filter=True, char_filter=False)
    print("Train")
    train_model(model, x_s_train, y_train, x_s_validation, y_validation, model_name, path="unk_exp/")
    print("Test-Normal")
    test_model(model, model_name, x_s_test_normal, y_test_normal, path="unk_exp/")
    print("Test-UNK-WORDS")
    test_model(model, model_name, x_s_test_unk_w, y_test_unk_w, path="unk_exp/")
    print("Test-UNK-CHAR")
    test_model(model, model_name, x_s_test_unk_c, y_test_unk_c, path="unk_exp/")

    model_name = "Radical-CNN-RNN HC"
    print("======MODEL: ", model_name, "======")
    model = build_sentence_rnn(real_vocab_number=real_vocab_number, char_vocab_size=char_vocab_size,
                               word_vocab_size=word_vocab_size, classes=num_class,
                               char_shape=True, word=False, char=False,
                               cnn_encoder=True, highway="relu", nohighway="linear",
                               attention=False, shape_filter=False, char_filter=True)
    print("Train")
    train_model(model, x_s_train, y_train, x_s_validation, y_validation, model_name, path="unk_exp/")
    print("Test-Normal")
    test_model(model, model_name, x_s_test_normal, y_test_normal, path="unk_exp/")
    print("Test-UNK-WORDS")
    test_model(model, model_name, x_s_test_unk_w, y_test_unk_w, path="unk_exp/")
    print("Test-UNK-CHAR")
    test_model(model, model_name, x_s_test_unk_c, y_test_unk_c, path="unk_exp/")

    model_name = "Radical-CNN-RNN RC"
    print("======MODEL: ", model_name, "======")
    model = build_sentence_rnn(real_vocab_number=real_vocab_number, char_vocab_size=char_vocab_size,
                               word_vocab_size=word_vocab_size, classes=num_class,
                               char_shape=True, word=False, char=False,
                               cnn_encoder=True, highway=None, nohighway="linear",
                               attention=False, shape_filter=True, char_filter=True)
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
    model = build_sentence_rnn(real_vocab_number=real_vocab_number, char_vocab_size=char_vocab_size,
                               word_vocab_size=word_vocab_size, classes=num_class,
                               char_shape=False, word=False, char=True,
                               cnn_encoder=True, highway="relu", nohighway="linear",
                               attention=True, shape_filter=True, char_filter=True)
    print("Train")
    train_model(model, x_c_train, y_train, x_c_validation, y_validation, model_name, path="unk_exp/")
    print("Test-Normal")
    test_model(model, model_name, x_c_test_normal, y_test_normal, path="unk_exp/")
    print("Test-UNK-WORDS")
    test_model(model, model_name, x_c_test_unk_w, y_test_unk_w, path="unk_exp/")
    print("Test-UNK-CHAR")
    test_model(model, model_name, x_c_test_unk_c, y_test_unk_c, path="unk_exp/")
    #
    model_name = "WORD-RNN"
    print("======MODEL: ", model_name, "======")
    model = build_sentence_rnn(real_vocab_number=real_vocab_number, char_vocab_size=char_vocab_size,
                               word_vocab_size=word_vocab_size, classes=num_class,
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
    #
    model_name = "WORD-HATT"
    print("======MODEL: ", model_name, "======")
    model = build_hatt(word_vocab_size, 2)
    print("Train")
    _x_w_train = numpy.reshape(x_w_train, (x_w_train.shape[0], 5, 100))
    _x_w_validation = numpy.reshape(x_w_validation, (x_w_validation.shape[0], 5, 100))
    _x_w_test_normal = numpy.reshape(x_w_test_normal, (x_w_test_normal.shape[0], 5, 100))
    _x_w_test_unk_w = numpy.reshape(x_w_test_unk_w, (x_w_test_unk_w.shape[0], 5, 100))
    _x_w_test_unk_c = numpy.reshape(x_w_test_unk_c, (x_w_test_unk_c.shape[0], 5, 100))
    train_model(model, _x_w_train, y_train, _x_w_validation, y_validation, model_name, path="unk_exp/")
    print("Test-Normal")
    test_model(model, model_name, _x_w_test_normal, y_test_normal, path="unk_exp/")
    print("Test-UNK-WORDS")
    test_model(model, model_name, _x_w_test_unk_w, y_test_unk_w, path="unk_exp/")
    print("Test-UNK-CHAR")
    test_model(model, model_name, _x_w_test_unk_c, y_test_unk_c, path="unk_exp/")

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

def analyse_dataset(s, full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin,
                  janome_tokenizer):

    n_hira_punc_number_latin = len(hira_punc_number_latin) + 2
    set_radical_vocab = set()
    char_vocab = set()
    word_vocab = set()
    kanji_count = 0
    char_count = 0
    word_count = 0

    preprocessed_char_number = len(full_vocab)

    positive = s["positive"]
    negative = s["negative"]

    for i, text in enumerate(tqdm(positive + negative)):
        # 日语分词
        janome = True
        parse_tokens = janome_tokenizer.tokenize(text)
        for j, mrph in enumerate(parse_tokens):
            if j + 1 > MAX_SENTENCE_LENGTH:
                break
            word_count += 1
            if janome:
                word = mrph.surface
            else:
                word = mrph.midasi
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
            word_vocab.add(word)
            for l, char_g in enumerate(word):
                char_count += 1
                if char_g in full_vocab:
                    if n_hira_punc_number_latin < full_vocab.index(char_g) < preprocessed_char_number:
                        kanji_count += 1
                char_vocab.add(char_g)
            # char shape level
            char_index = text_to_char_index(full_vocab=full_vocab, real_vocab_number=real_vocab_number,
                                            chara_bukken_revised=chara_bukken_revised,
                                            addition_translate=additional_translate,
                                            sentence_text=word, preprocessed_char_number=preprocessed_char_number,
                                            skip_unknown=False, shuffle=None)
            for int in char_index:
                set_radical_vocab.add(int)
    print("radical vocab size: ", len(set_radical_vocab))
    print("char vocab size: ", len(char_vocab))
    print("word vocab size: ", len(word_vocab))
    print("kanji count: ", kanji_count)
    print("char count: ", char_count)
    print("word count: ", word_count)


def analyse_datasets():
    janome_tokenizer = JanomeTokenizer()
    train_set, tune_set, validation_set, test_normal_set, test_unk_w_set, test_unk_c_set \
        = load_data()
    full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin = get_vocab()
    analyse_dataset(train_set, full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin, janome_tokenizer)
    analyse_dataset(tune_set, full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin, janome_tokenizer)
    analyse_dataset(validation_set, full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin, janome_tokenizer)
    analyse_dataset(test_normal_set, full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin, janome_tokenizer)
    analyse_dataset(test_unk_w_set, full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin, janome_tokenizer)
    analyse_dataset(test_unk_c_set, full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin, janome_tokenizer)


def lime_radical_classifier_arc_wordI(string_list):
    janome_tokenizer = JanomeTokenizer()
    x_data = numpy.zeros((len(string_list), MAX_SENTENCE_LENGTH, COMP_WIDTH * MAX_WORD_LENGTH))
    full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin = get_vocab()
    n_hira_punc_number_latin = len(hira_punc_number_latin) + 2
    for i, text in enumerate(string_list):
        parse_tokens = janome_tokenizer.tokenize(text)
        for j, mrph in enumerate(parse_tokens):
            if j + 1 > MAX_SENTENCE_LENGTH:
                break
            word = mrph.surface
            char_index = text_to_char_index(full_vocab=full_vocab, real_vocab_number=real_vocab_number,
                                                chara_bukken_revised=chara_bukken_revised,
                                                addition_translate=additional_translate,
                                                sentence_text=word, preprocessed_char_number=len(full_vocab),
                                                skip_unknown=False, shuffle=None)
            if len(char_index) < COMP_WIDTH * MAX_WORD_LENGTH:
                char_index = char_index + [0] * (COMP_WIDTH * MAX_WORD_LENGTH - len(char_index))  # Padding
            elif len(char_index) > COMP_WIDTH * MAX_WORD_LENGTH:
                char_index = char_index[:COMP_WIDTH * MAX_WORD_LENGTH]
            for k, comp in enumerate(char_index):
                if k < COMP_WIDTH * MAX_WORD_LENGTH:
                    x_data[i, j, k] = comp
    model_name = "Radical-CNN-RNN ARC"
    print("======MODEL: ", model_name, "======")
    model = build_sentence_rnn(real_vocab_number=real_vocab_number, classes=2,
                               char_shape=True, word=False, char=False,
                               cnn_encoder=True, highway=None, nohighway="linear",
                               attention=True, shape_filter=True, char_filter=True)
    model.load_weights("unk_exp/checkpoints/" + model_name + "_bestloss.hdf5")
    predicts = model.predict(x_data)
    y_output = numpy.zeros((len(string_list), 2), dtype=numpy.float32)
    for i, predicts in enumerate(predicts):
        y_output[i] = predicts
    return y_output

def lime_radical_harc_classifier_wordI(string_list):
    janome_tokenizer = JanomeTokenizer()
    x_data = numpy.zeros((len(string_list), MAX_SENTENCE_LENGTH, COMP_WIDTH * MAX_WORD_LENGTH))
    full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin = get_vocab()
    n_hira_punc_number_latin = len(hira_punc_number_latin) + 2
    for i, text in enumerate(string_list):
        parse_tokens = janome_tokenizer.tokenize(text)
        for j, mrph in enumerate(parse_tokens):
            if j + 1 > MAX_SENTENCE_LENGTH:
                break
            word = mrph.surface
            char_index = text_to_char_index(full_vocab=full_vocab, real_vocab_number=real_vocab_number,
                                                chara_bukken_revised=chara_bukken_revised,
                                                addition_translate=additional_translate,
                                                sentence_text=word, preprocessed_char_number=len(full_vocab),
                                                skip_unknown=False, shuffle=None)
            if len(char_index) < COMP_WIDTH * MAX_WORD_LENGTH:
                char_index = char_index + [0] * (COMP_WIDTH * MAX_WORD_LENGTH - len(char_index))  # Padding
            elif len(char_index) > COMP_WIDTH * MAX_WORD_LENGTH:
                char_index = char_index[:COMP_WIDTH * MAX_WORD_LENGTH]
            for k, comp in enumerate(char_index):
                if k < COMP_WIDTH * MAX_WORD_LENGTH:
                    x_data[i, j, k] = comp
    model_name = "Radical-CNN-RNN HARC"
    print("======MODEL: ", model_name, "======")
    model = build_sentence_rnn(real_vocab_number=real_vocab_number, classes=2,
                               char_shape=True, word=False, char=False,
                               cnn_encoder=True, highway='relu', nohighway="linear",
                               attention=True, shape_filter=True, char_filter=True)
    model.load_weights("unk_exp/checkpoints/" + model_name + "_bestloss.hdf5")
    predicts = model.predict(x_data)
    y_output = numpy.zeros((len(string_list), 2), dtype=numpy.float32)
    for i, predicts in enumerate(predicts):
        y_output[i] = predicts
    return y_output

def lime_radical_rc_classifier_wordI(string_list):
    janome_tokenizer = JanomeTokenizer()
    x_data = numpy.zeros((len(string_list), MAX_SENTENCE_LENGTH, COMP_WIDTH * MAX_WORD_LENGTH))
    full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin = get_vocab()
    n_hira_punc_number_latin = len(hira_punc_number_latin) + 2
    for i, text in enumerate(string_list):
        parse_tokens = janome_tokenizer.tokenize(text)
        for j, mrph in enumerate(parse_tokens):
            if j + 1 > MAX_SENTENCE_LENGTH:
                break
            word = mrph.surface
            char_index = text_to_char_index(full_vocab=full_vocab, real_vocab_number=real_vocab_number,
                                                chara_bukken_revised=chara_bukken_revised,
                                                addition_translate=additional_translate,
                                                sentence_text=word, preprocessed_char_number=len(full_vocab),
                                                skip_unknown=False, shuffle=None)
            if len(char_index) < COMP_WIDTH * MAX_WORD_LENGTH:
                char_index = char_index + [0] * (COMP_WIDTH * MAX_WORD_LENGTH - len(char_index))  # Padding
            elif len(char_index) > COMP_WIDTH * MAX_WORD_LENGTH:
                char_index = char_index[:COMP_WIDTH * MAX_WORD_LENGTH]
            for k, comp in enumerate(char_index):
                if k < COMP_WIDTH * MAX_WORD_LENGTH:
                    x_data[i, j, k] = comp
    model_name = "Radical-CNN-RNN RC"
    print("======MODEL: ", model_name, "======")
    model = build_sentence_rnn(real_vocab_number=real_vocab_number, classes=2,
                               char_shape=True, word=False, char=False,
                               cnn_encoder=True, highway=None, nohighway="linear",
                               attention=False, shape_filter=True, char_filter=True)
    model.load_weights("unk_exp/checkpoints/" + model_name + "_bestloss.hdf5")
    predicts = model.predict(x_data)
    y_output = numpy.zeros((len(string_list), 2), dtype=numpy.float32)
    for i, predicts in enumerate(predicts):
        y_output[i] = predicts
    return y_output

def lime_radical_har_classifier_wordI(string_list):
    janome_tokenizer = JanomeTokenizer()
    x_data = numpy.zeros((len(string_list), MAX_SENTENCE_LENGTH, COMP_WIDTH * MAX_WORD_LENGTH))
    full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin = get_vocab()
    n_hira_punc_number_latin = len(hira_punc_number_latin) + 2
    for i, text in enumerate(string_list):
        parse_tokens = janome_tokenizer.tokenize(text)
        for j, mrph in enumerate(parse_tokens):
            if j + 1 > MAX_SENTENCE_LENGTH:
                break
            word = mrph.surface
            char_index = text_to_char_index(full_vocab=full_vocab, real_vocab_number=real_vocab_number,
                                                chara_bukken_revised=chara_bukken_revised,
                                                addition_translate=additional_translate,
                                                sentence_text=word, preprocessed_char_number=len(full_vocab),
                                                skip_unknown=False, shuffle=None)
            if len(char_index) < COMP_WIDTH * MAX_WORD_LENGTH:
                char_index = char_index + [0] * (COMP_WIDTH * MAX_WORD_LENGTH - len(char_index))  # Padding
            elif len(char_index) > COMP_WIDTH * MAX_WORD_LENGTH:
                char_index = char_index[:COMP_WIDTH * MAX_WORD_LENGTH]
            for k, comp in enumerate(char_index):
                if k < COMP_WIDTH * MAX_WORD_LENGTH:
                    x_data[i, j, k] = comp
    model_name = "Radical-CNN-RNN HAR"
    print("======MODEL: ", model_name, "======")
    model = build_sentence_rnn(real_vocab_number=real_vocab_number, classes=2,
                               char_shape=True, word=False, char=False,
                               cnn_encoder=True, highway='relu', nohighway="linear",
                               attention=True, shape_filter=True, char_filter=False)
    model.load_weights("unk_exp/checkpoints/" + model_name + "_bestloss.hdf5")
    predicts = model.predict(x_data)
    y_output = numpy.zeros((len(string_list), 2), dtype=numpy.float32)
    for i, predicts in enumerate(predicts):
        y_output[i] = predicts
    return y_output

def lime_radical_hac_classifier_wordI(string_list):
    janome_tokenizer = JanomeTokenizer()
    x_data = numpy.zeros((len(string_list), MAX_SENTENCE_LENGTH, COMP_WIDTH * MAX_WORD_LENGTH))
    full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin = get_vocab()
    n_hira_punc_number_latin = len(hira_punc_number_latin) + 2
    for i, text in enumerate(string_list):
        parse_tokens = janome_tokenizer.tokenize(text)
        for j, mrph in enumerate(parse_tokens):
            if j + 1 > MAX_SENTENCE_LENGTH:
                break
            word = mrph.surface
            char_index = text_to_char_index(full_vocab=full_vocab, real_vocab_number=real_vocab_number,
                                                chara_bukken_revised=chara_bukken_revised,
                                                addition_translate=additional_translate,
                                                sentence_text=word, preprocessed_char_number=len(full_vocab),
                                                skip_unknown=False, shuffle=None)
            if len(char_index) < COMP_WIDTH * MAX_WORD_LENGTH:
                char_index = char_index + [0] * (COMP_WIDTH * MAX_WORD_LENGTH - len(char_index))  # Padding
            elif len(char_index) > COMP_WIDTH * MAX_WORD_LENGTH:
                char_index = char_index[:COMP_WIDTH * MAX_WORD_LENGTH]
            for k, comp in enumerate(char_index):
                if k < COMP_WIDTH * MAX_WORD_LENGTH:
                    x_data[i, j, k] = comp
    model_name = "Radical-CNN-RNN HAC"
    print("======MODEL: ", model_name, "======")
    model = build_sentence_rnn(real_vocab_number=real_vocab_number, classes=2,
                               char_shape=True, word=False, char=False,
                               cnn_encoder=True, highway='relu', nohighway="linear",
                               attention=True, shape_filter=False, char_filter=True)
    model.load_weights("unk_exp/checkpoints/" + model_name + "_bestloss.hdf5")
    predicts = model.predict(x_data)
    y_output = numpy.zeros((len(string_list), 2), dtype=numpy.float32)
    for i, predicts in enumerate(predicts):
        y_output[i] = predicts
    return y_output

def lime_character_classifier_wordI(string_list):
    janome_tokenizer = JanomeTokenizer()
    x_data = numpy.zeros((len(string_list), MAX_SENTENCE_LENGTH, MAX_WORD_LENGTH))
    full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin = get_vocab()
    n_hira_punc_number_latin = len(hira_punc_number_latin) + 2
    with open("unk_exp/rakuten_processed_review_split.pickle", "rb") as f:
        full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin, \
        preprocessed_char_number, word_vocab, char_vocab, \
        x_s_train, x_c_train, x_w_train, y_train, \
        x_s_validation, x_c_validation, x_w_validation, y_validation, \
        x_s_test_normal, x_c_test_normal, x_w_test_normal, y_test_normal, \
        x_s_test_unk_w, x_c_test_unk_w, x_w_test_unk_w, y_test_unk_w, \
        x_s_test_unk_c, x_c_test_unk_c, x_w_test_unk_c, y_test_unk_c = pickle.load(f)
    word_vocab_size = len(word_vocab)
    char_vocab_size = len(char_vocab)
    for i, text in enumerate(string_list):
        parse_tokens = janome_tokenizer.tokenize(text)
        for j, mrph in enumerate(parse_tokens):
            if j + 1 > MAX_SENTENCE_LENGTH:
                break
            word = mrph.surface
            for k, char in enumerate(word):
                if char not in char_vocab:
                    char_vocab.append(char)
                    char_g_index = len(char_vocab) - 1
                else:
                    char_g_index = char_vocab.index(char)
                if k < MAX_WORD_LENGTH:
                    x_data[i, j, k] = char_g_index
    model_name = "CHAR-CNN-RNN"
    print("======MODEL: ", model_name, "======")
    model = build_sentence_rnn(real_vocab_number=real_vocab_number, char_vocab_size=char_vocab_size,
                               word_vocab_size=word_vocab_size, classes=2,
                               char_shape=False, word=False, char=True,
                               cnn_encoder=True, highway="relu", nohighway="linear",
                               attention=True, shape_filter=True, char_filter=True)
    model.load_weights("unk_exp/checkpoints/" + model_name + "_bestloss.hdf5")
    predicts = model.predict(x_data)
    y_output = numpy.zeros((len(string_list), 2), dtype=numpy.float32)
    for i, predicts in enumerate(predicts):
        y_output[i] = predicts
    return y_output

def lime_fasttext_classifier_wordI(string_list):
    janome_tokenizer = JanomeTokenizer()
    x_data = numpy.zeros((len(string_list), MAX_SENTENCE_LENGTH))
    full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin = get_vocab()
    n_hira_punc_number_latin = len(hira_punc_number_latin) + 2
    with open("unk_exp/rakuten_processed_review_split.pickle", "rb") as f:
        full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin, \
        preprocessed_char_number, word_vocab, char_vocab, \
        x_s_train, x_c_train, x_w_train, y_train, \
        x_s_validation, x_c_validation, x_w_validation, y_validation, \
        x_s_test_normal, x_c_test_normal, x_w_test_normal, y_test_normal, \
        x_s_test_unk_w, x_c_test_unk_w, x_w_test_unk_w, y_test_unk_w, \
        x_s_test_unk_c, x_c_test_unk_c, x_w_test_unk_c, y_test_unk_c = pickle.load(f)
    word_vocab_size = len(word_vocab)
    char_vocab_size = len(char_vocab)
    for i, text in enumerate(string_list):
        parse_tokens = janome_tokenizer.tokenize(text)
        for j, mrph in enumerate(parse_tokens):
            if j + 1 > MAX_SENTENCE_LENGTH:
                break
            word = mrph.surface
            if word not in word_vocab:
                word_vocab.append(word)
                word_index = len(word_vocab) - 1
            else:
                word_index = word_vocab.index(word)
            x_data[i, j] = word_index
    model_name = "WORD-FASTTEXT"
    print("======MODEL: ", model_name, "======")
    model = build_fasttext(word_vocab_size, 2)
    model.load_weights("unk_exp/checkpoints/" + model_name + "_bestloss.hdf5")
    predicts = model.predict(x_data)
    y_output = numpy.zeros((len(string_list), 2), dtype=numpy.float32)
    for i, predicts in enumerate(predicts):
        y_output[i] = predicts
    return y_output

def lime_analyse(input_text):
    explainer = LimeTextExplainer(class_names = ["negative", "positive"])
    # exp = explainer.explain_instance(input_text, lime_radical_classifier_wordI)
    # exp.save_to_file('radical_arc_oi.html')
    # exp = explainer.explain_instance(input_text, lime_character_classifier_wordI)
    # exp.save_to_file('character_oi.html')
    # exp = explainer.explain_instance(input_text, lime_fasttext_classifier_wordI)
    # exp.save_to_file('fasttext_oi.html')
    exp = explainer.explain_instance(input_text, lime_radical_rc_classifier_wordI)
    exp.save_to_file('radical_rc_oi.html')
    exp = explainer.explain_instance(input_text, lime_radical_har_classifier_wordI)
    exp.save_to_file('radical_har_oi.html')
    exp = explainer.explain_instance(input_text, lime_radical_hac_classifier_wordI)
    exp.save_to_file('radical_hac_oi.html')


def lime_find_good_example(texts, true_label):
    y_outputs_radical = lime_radical_harc_classifier_wordI(texts)
    y_outputs_canlm = lime_character_classifier_wordI(texts)
    y_outputs_fast = lime_fasttext_classifier_wordI(texts)
    for i, outputs in enumerate(zip(y_outputs_radical, y_outputs_canlm, y_outputs_fast)):
        if outputs[0][true_label] > 0.5 > outputs[1][true_label] and outputs[2][true_label] < 0.5:
            print("Good example: #", i, ": ", texts[i])
            input_text = texts[i]
            explainer = LimeTextExplainer(class_names=["negative", "positive"])
            exp = explainer.explain_instance(input_text, lime_radical_harc_classifier_wordI)
            fig = exp.as_pyplot_figure()
            exp.save_to_file('radical_oi.html')
            exp = explainer.explain_instance(input_text, lime_character_classifier_wordI)
            fig = exp.as_pyplot_figure()
            exp.save_to_file('character_oi.html')
            exp = explainer.explain_instance(input_text, lime_fasttext_classifier_wordI)
            fig = exp.as_pyplot_figure()
            exp.save_to_file('fasttext_oi.html')
            break


def lime_find_wrong_output(test_sets):
    count = 0
    for test_set in test_sets:
        y_outputs_radical = lime_radical_harc_classifier_wordI(test_set["positive"])
        for text, predict in zip(test_set["positive"], y_outputs_radical):
            if predict[0] > 0.5:
                count += 1
                explainer = LimeTextExplainer(class_names=["negative", "positive"])
                exp = explainer.explain_instance(text, lime_radical_harc_classifier_wordI)
                exp.save_to_file('LIME_error_analysis/radical_oi_'+str(count)+'_positive.html')
        y_outputs_radical = lime_radical_harc_classifier_wordI(test_set["negative"])
        for text, predict in zip(test_set["negative"], y_outputs_radical):
            if predict[1] > 0.5:
                count += 1
                explainer = LimeTextExplainer(class_names=["negative", "positive"])
                exp = explainer.explain_instance(text, lime_radical_harc_classifier_wordI)
                exp.save_to_file('LIME_error_analysis/radical_oi_' + str(count) + '_negative.html')


if __name__ == "__main__":
    # unk_exp_preproces_j()
    unk_experiment_j_p1()
    # unk_experiment_j_p2()
    # analyse_datasets()

    # train_set, tune_set, validation_set, test_normal_set, test_unk_w_set, test_unk_c_set \
    #     = pickle.load(open("rakuten/rakuten_review_split.pickle", "rb"))
    # lime_analyse(test_unk_w_set["positive"][80])
    # lime_find_good_example(test_unk_w_set["positive"], 1)
    # lime_find_wrong_output([test_normal_set, test_unk_w_set, test_unk_c_set])
