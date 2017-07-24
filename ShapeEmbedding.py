import re, string, pickle, numpy, pandas, mojimoji, datetime, os, jieba, sys
from pyknp import Jumanpp
from keras import optimizers
from keras.models import Model
from keras.layers import Embedding, Input, AveragePooling1D, MaxPooling1D, Conv1D, concatenate, TimeDistributed, \
    Bidirectional, LSTM, Dense, Flatten
from keras.legacy.layers import Highway
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from attention import AttentionWithContext
from getShapeCode import get_all_word_bukken

# MAX_SENTENCE_LENGTH = 739  # large number as 739 makes cudnn die
MAX_SENTENCE_LENGTH = 500
MAX_WORD_LENGTH = 3
COMP_WIDTH = 3
CHAR_EMB_DIM = 15
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 100
WORD_DIM = 150


def _make_kana_convertor():
    # by http://d.hatena.ne.jp/mohayonao/20091129/1259505966
    """ひらがな⇔カタカナ変換器を作る"""
    kata = {
        'ア': 'あ', 'イ': 'い', 'ウ': 'う', 'エ': 'え', 'オ': 'お',
        'カ': 'か', 'キ': 'き', 'ク': 'く', 'ケ': 'け', 'コ': 'こ',
        'サ': 'さ', 'シ': 'し', 'ス': 'す', 'セ': 'せ', 'ソ': 'そ',
        'タ': 'た', 'チ': 'ち', 'ツ': 'つ', 'テ': 'て', 'ト': 'と',
        'ナ': 'な', 'ニ': 'に', 'ヌ': 'ぬ', 'ネ': 'ね', 'ノ': 'の',
        'ハ': 'は', 'ヒ': 'ひ', 'フ': 'ふ', 'ヘ': 'へ', 'ホ': 'ほ',
        'マ': 'ま', 'ミ': 'み', 'ム': 'む', 'メ': 'め', 'モ': 'も',
        'ヤ': 'や', 'ユ': 'ゆ', 'ヨ': 'よ', 'ラ': 'ら', 'リ': 'り',
        'ル': 'る', 'レ': 'れ', 'ロ': 'ろ', 'ワ': 'わ', 'ヲ': 'を',
        'ン': 'ん',

        'ガ': 'が', 'ギ': 'ぎ', 'グ': 'ぐ', 'ゲ': 'げ', 'ゴ': 'ご',
        'ザ': 'ざ', 'ジ': 'じ', 'ズ': 'ず', 'ゼ': 'ぜ', 'ゾ': 'ぞ',
        'ダ': 'だ', 'ヂ': 'ぢ', 'ヅ': 'づ', 'デ': 'で', 'ド': 'ど',
        'バ': 'ば', 'ビ': 'び', 'ブ': 'ぶ', 'ベ': 'べ', 'ボ': 'ぼ',
        'パ': 'ぱ', 'ピ': 'ぴ', 'プ': 'ぷ', 'ペ': 'ぺ', 'ポ': 'ぽ',

        'ァ': 'ぁ', 'ィ': 'ぃ', 'ゥ': 'ぅ', 'ェ': 'ぇ', 'ォ': 'ぉ',
        'ャ': 'ゃ', 'ュ': 'ゅ', 'ョ': 'ょ',
        'ッ': 'っ', 'ヰ': 'ゐ', 'ヱ': 'ゑ',
    }

    # ひらがな → カタカナ のディクショナリをつくる
    hira = dict([(v, k) for k, v in kata.items()])

    re_hira2kata = re.compile("|".join(map(re.escape, hira)))
    re_kata2hira = re.compile("|".join(map(re.escape, kata)))

    def _hiragana2katakana(text):
        return re_hira2kata.sub(lambda x: hira[x.group(0)], text)

    def _katakana2hiragana(text):
        return re_kata2hira.sub(lambda x: kata[x.group(0)], text)

    return (_hiragana2katakana, _katakana2hiragana, hira.keys())


def load_shape_data(datafile="usc-shape_bukken_data.pickle"):
    with open(datafile) as f:
        data = pickle.load(f)
    return data["words"], data["bukkens"], data["word_bukken"]


def get_vocab(opts=None):
    # convert kata to hira
    char_emb_dim = CHAR_EMB_DIM
    use_component = True  # True for component level False for chara level

    _, _, hirakana_list = _make_kana_convertor()
    addition_translate = str.maketrans("ッャュョヮヵヶ?？⁇⁈⁉﹗!‼！″＂“”『』「」‘’´｀:;。、・"
                                       "＼([｛)]｝〔〕【〘〖】〙〗｟｠«»ー－—–‐゠〜～〳〵￥",
                                       "っゃゅょゎゕゖ?????!!!!\"\"\"\"\"\"'''''':;.,･"
                                       "\\((()))()((()))()《》-----=~~/\\$")

    hira_punc_number_latin = "".join(hirakana_list) + string.punctuation + \
                             'ヴゎゕゖㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇷ゚ㇺㇻㇼㇽㇾㇿ々〻' \
                             '〟ゝゞ〈《〉》〝…‥〴' \
                             '1234567890' \
                             'abcdefghijklmnopqrstuvwxyz ' \
                             '○●☆★■♪ヾω*≧∇≦※→←↑↓'
    # note: the space and punctuations in Jp sometimes show emotion

    vocab_chara, vocab_bukken, chara_bukken = get_all_word_bukken()
    hira_punc_number_latin_number = len(hira_punc_number_latin) + 2
    print("totally {n} kana, punctuation and latin char".format(n=str(hira_punc_number_latin_number)))
    vocab = ["</padblank>", "</s>"] + list(hira_punc_number_latin) + vocab_bukken
    real_vocab_number = len(vocab)  # the part of the vocab that is really used. only basic components
    vocab_chara_strip = [chara for chara in vocab_chara if chara not in vocab_bukken]  # delete 独体字
    print("totally {n} puctuation, kana, latin, and chara components".format(n=str(real_vocab_number)))
    full_vocab = vocab + vocab_chara_strip  # add unk at the head, and complex charas for text encoding at the tail
    chara_bukken_revised = {}
    for i_word, i_bukken in chara_bukken.items():  # update the index
        if vocab_chara[i_word] not in vocab_bukken:  # delete 独体字
            chara_bukken_revised[full_vocab.index(vocab_chara[i_word])] = \
                [k + hira_punc_number_latin_number for k in i_bukken]
    del vocab_chara
    del chara_bukken

    return full_vocab, real_vocab_number, chara_bukken_revised, addition_translate, hira_punc_number_latin


def text_to_char_index(full_vocab, real_vocab_number, chara_bukken_revised, sentence_text, addition_translate,
                        mode="padding", comp_width=COMP_WIDTH):
    # mode:
    # average: will repeat the original index to #comp_width for the process of the embedding layer
    # padding: will pad the original index to #comp_width with zero for the process of the embedding layer
    # char_emb_dim  char embedding size
    # comp_width  #components used

    # convert digital number and latin to hangaku
    text = mojimoji.zen_to_han(sentence_text, kana=False)
    # convert kana to zengaku
    text = mojimoji.han_to_zen(text, digit=False, ascii=False)
    # convert kata to hira
    _, katakana2hiragana, _ = _make_kana_convertor()
    text = katakana2hiragana(text)
    text = text.translate(addition_translate)
    # finally, lowercase
    text = text.lower()
    # expanding every character with 3 components
    ch2id = {}
    for i, w in enumerate(full_vocab):
        ch2id[w] = i
    int_text = []
    # print(text)
    if mode == "average":
        for c in text:
            try:
                i = ch2id[c]
            except KeyError:
                continue
            if i > real_vocab_number:
                comps = chara_bukken_revised[i]
                if len(comps) >= comp_width:
                    int_text += comps[:comp_width]
                elif len(comps) == 1:
                    int_text += [i] * comp_width
                else:
                    int_text += comps + [0] * (comp_width - len(comps))
            else:
                int_text += [i] * comp_width
    elif mode == "padding":
        for c in text:
            # print(c)
            try:
                i = ch2id[c]
            except KeyError:
                continue
            # print(i)
            if i > real_vocab_number:
                comps = chara_bukken_revised[i]
                # print(comps)
                if len(comps) >= comp_width:
                    int_text += comps[:comp_width]
                else:
                    int_text += comps + [0] * (comp_width - len(comps))
            else:
                int_text += [i] + [0] * (comp_width - 1)
    return int_text


def build_word_feature_shape(vocab_size=5, char_emb_dim=CHAR_EMB_DIM, comp_width=COMP_WIDTH,
                       mode="padding", cnn_encoder=True):
    # build the feature computed by cnn for each word in the sentence. used to input to the next rnn.
    # expected input: every #comp_width int express a character.
    # mode:
    # "average": average pool the every #comp_with input embedding, output average of the indexed embeddings of a character
    # "padding": convoluate every #comp_width embedding

    # real vocab_size for ucs is 2481, including paddingblank, unkown, puncutations, kanas
    init_width = 0.5 / char_emb_dim
    init_weight = numpy.random.uniform(low=-init_width, high=init_width, size=(vocab_size, char_emb_dim))
    init_weight[0] = 0  # maybe the padding should not be zero
    # print(init_weight)
    # first layer embeds
    #  every components
    word_input = Input(shape=(COMP_WIDTH * MAX_WORD_LENGTH,))
    char_embedding = \
        Embedding(input_dim=vocab_size, output_dim=char_emb_dim, weights=[init_weight], trainable=True)(word_input)
    # print("char_embedding:", char_embedding._keras_shape)
    if cnn_encoder:
        if mode == "average":
            # 2nd layer average the #comp_width components of every character
            char_embedding = AveragePooling1D(pool_size=comp_width, strides=comp_width, padding='valid')
            # conv, filter width 1 2 3, feature maps 50 100 150
            feature1 = Conv1D(filters=50, kernel_size=1, activation='relu')(char_embedding)
            feature1 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 1 + 1)(feature1)
            feature2 = Conv1D(filters=100, kernel_size=2, activation='relu')(char_embedding)
            feature2 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 2 + 1)(feature2)
            feature3 = Conv1D(filters=150, kernel_size=3, activation='relu')(char_embedding)
            feature3 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 3 + 1)(feature3)
            feature = concatenate([feature1, feature2, feature3])
        elif mode == "padding":
            # print(char_embedding._keras_shape)
            # conv, filter with [1, 2, 3]*#comp_width, feature maps 50 100 150
            print(comp_width)
            feature1 = Conv1D(filters=50, kernel_size=1 * comp_width, strides=comp_width, activation='relu')(
                char_embedding)
            feature1 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 1 + 1)(feature1)
            feature2 = Conv1D(filters=100, kernel_size=2 * comp_width, strides=comp_width, activation='relu')(
                char_embedding)
            feature2 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 2 + 1)(feature2)
            feature3 = Conv1D(filters=150, kernel_size=3 * comp_width, strides=comp_width, activation='relu')(
                char_embedding)
            feature3 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 3 + 1)(feature3)
            feature = concatenate([feature1, feature2, feature3])
        feature = Flatten()(feature)
        # print(feature._keras_shape)
        feature = Highway()(feature)
    else:
        feature = Flatten()(char_embedding)
    word_feature_encoder = Model(word_input, feature)
    return word_feature_encoder

def build_word_feature_char(vocab_size=5, char_emb_dim=CHAR_EMB_DIM,
                       mode="padding", cnn_encoder=True):
    # build the feature computed by cnn for each word in the sentence. used to input to the next rnn.
    # expected input: every #comp_width int express a character.
    # mode:
    # "average": average pool the every #comp_with input embedding, output average of the indexed embeddings of a character
    # "padding": convoluate every #comp_width embedding

    # real vocab_size for ucs is 2481, including paddingblank, unkown, puncutations, kanas
    init_width = 0.5 / char_emb_dim
    init_weight = numpy.random.uniform(low=-init_width, high=init_width, size=(vocab_size, char_emb_dim))
    init_weight[0] = 0  # maybe the padding should not be zero
    # print(init_weight)
    # first layer embeds
    #  every components
    word_input = Input(shape=(MAX_WORD_LENGTH,))
    char_embedding = \
        Embedding(input_dim=vocab_size, output_dim=char_emb_dim, weights=[init_weight], trainable=True)(word_input)
    # print("char_embedding:", char_embedding._keras_shape)
    if cnn_encoder:
        if mode == "padding":
            # print(char_embedding._keras_shape)
            # conv, filter with [1, 2, 3]*#comp_width, feature maps 50 100 150
            feature1 = Conv1D(filters=50, kernel_size=1, activation='relu')(
                char_embedding)
            feature1 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 1 + 1)(feature1)
            feature2 = Conv1D(filters=100, kernel_size=2, activation='relu')(
                char_embedding)
            feature2 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 2 + 1)(feature2)
            feature3 = Conv1D(filters=150, kernel_size=3, activation='relu')(
                char_embedding)
            feature3 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 3 + 1)(feature3)
            feature = concatenate([feature1, feature2, feature3])
        feature = Flatten()(feature)
        # print(feature._keras_shape)
        feature = Highway()(feature)
    else:
        feature = Flatten()(char_embedding)
    word_feature_encoder = Model(word_input, feature)
    return word_feature_encoder

def build_sentence_rnn(real_vocab_number, word_vocab_size=10, char_vocab_size=10,
                       classes=2, attention=False, dropout=0,
                       word=True, char=False, char_shape=True, model="rnn", cnn_encoder=True):
    print(MAX_SENTENCE_LENGTH)
    # build the rnn of words, use the output of build_word_feature as the feature of each word
    if char_shape:
        word_feature_encoder = build_word_feature_shape(vocab_size=real_vocab_number, cnn_encoder=cnn_encoder)
        sentence_input = Input(shape=(MAX_SENTENCE_LENGTH, COMP_WIDTH * MAX_WORD_LENGTH), dtype='int32')
        word_feature_sequence = TimeDistributed(word_feature_encoder)(sentence_input)
        # print(word_feature_sequence._keras_shape)
    if word:
        sentence_word_input = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
        word_embedding_sequence = Embedding(input_dim=word_vocab_size, output_dim=WORD_DIM)(sentence_word_input)
    if char:
        word_feature_encoder = build_word_feature_char(vocab_size=char_vocab_size, cnn_encoder=cnn_encoder)
        char_input = Input(shape=(MAX_SENTENCE_LENGTH, MAX_WORD_LENGTH), dtype='int32')
        word_feature_sequence = TimeDistributed(word_feature_encoder)(char_input)
    if char_shape and word and not char:
        word_feature_sequence = concatenate([word_feature_sequence, word_embedding_sequence], axis=2)
    if word and not char_shape and not char:
        word_feature_sequence = word_embedding_sequence
    print(word_feature_sequence._keras_shape)
    if model == "rnn":
        if attention:
            lstm_rnn = Bidirectional(LSTM(150, dropout=dropout, return_sequences=True))(word_feature_sequence)
            lstm_rnn = TimeDistributed(Highway())(lstm_rnn)
            lstm_rnn = AttentionWithContext()(lstm_rnn)
        else:
            lstm_rnn = Bidirectional(LSTM(150, dropout=dropout, return_sequences=False))(word_feature_sequence)
        # print(lstm_rnn._keras_shape)
        # lstm_rnn = TimeDistributed(Highway())(lstm_rnn)
        x = lstm_rnn
    if classes < 2:
        print("class number cannot less than 2")
        exit(1)
    else:
        preds = Dense(classes, activation='softmax')(x)
    if char_shape and not word and not char:
        sentence_model = Model(sentence_input, preds)
    if word and not char_shape and not char:
        sentence_model = Model(sentence_word_input, preds)
    if word and char_shape and not char:
        sentence_model = Model([sentence_input, sentence_word_input], preds)
    if char and not word and not char_shape:
        sentence_model = Model(char_input, preds)
    sentence_model.summary()
    return sentence_model


def prepare_kyoto_classification(dev_mode=False, juman=True):
    # expected input. a sequence of the forward output of build_word_feature

    # get vocab
    full_vocab, real_vocab_number, chara_bukken_revised, addtional_translate,_ = get_vocab()

    # if isinstance(sentence_input_text, str):
    #     sentence_input_text = sentence_input_text.split()
    # sentence_input_int = [text_to_char_index(full_vocab=full_vocab, real_vocab_number=real_vocab_number,
    #                                         chara_bukken_revised=chara_bukken_revised,
    #                                         sentence_text=x) for x in sentence_input_text]
    # preprocess data

    # read xlsx
    data_f = pandas.ExcelFile('/home/yuanzhike/CMWE/kt_blog_data/EvaluativeInformationCorpus/EvalAnnotation-A.xlsx')
    sen_sheet = data_f.parse('Sheet1')
    rows, columns = sen_sheet.shape
    print("Totally {n} sentences in the dataset".format(n=rows))
    if dev_mode:
        max_rows = 5
    else:
        max_rows = rows
    sentence_texts = []
    labels_ng = []
    labels_14 = []
    label_14_names = ['メリット－', 'メリット＋', '感情－', '感情＋',
                      '採否－', '採否＋', '出来事－', '出来事＋',
                      '当為－', '当為＋', '批評－', '批評＋',
                      '要望－', '要望＋']
    for i in range(max_rows):
        sentence_texts.append(sen_sheet.iloc[i][2])
        if sen_sheet.iloc[i][3] == '+':
            labels_ng.append(1)
        else:
            labels_ng.append(0)
        labels_14.append(label_14_names.index(sen_sheet.iloc[i][5]))

    # change the sentence into matrix of word sequence
    data = numpy.zeros((max_rows, MAX_SENTENCE_LENGTH, COMP_WIDTH * MAX_WORD_LENGTH), dtype=numpy.int32)
    data_word = numpy.zeros((max_rows, MAX_SENTENCE_LENGTH), dtype=numpy.int32)
    print("Data shape: {shape}".format(shape=str(data.shape)))

    word_vocab = ["</s>"]

    if juman:
        juman = Jumanpp()
        for i, text in enumerate(sentence_texts):
            # print(text)
            parse_result = juman.analysis(text)
            for j, mrph in enumerate(parse_result.mrph_list()):
                if j + 1 > MAX_SENTENCE_LENGTH:
                    break
                word = mrph.genkei
                word_index = 0
                if word not in word_vocab:
                    word_vocab.append(word)
                    word_index = len(word_vocab) - 1
                else:
                    word_index = word_vocab.index(word)
                char_index = text_to_char_index(full_vocab=full_vocab, real_vocab_number=real_vocab_number,
                                                chara_bukken_revised=chara_bukken_revised, sentence_text=word,
                                                addition_translate=addtional_translate)
                if len(char_index) < COMP_WIDTH * MAX_WORD_LENGTH:
                    char_index = char_index + [0] * (COMP_WIDTH * MAX_WORD_LENGTH - len(char_index))  # Padding
                elif len(char_index) > COMP_WIDTH * MAX_WORD_LENGTH:
                    char_index = char_index[:COMP_WIDTH * MAX_WORD_LENGTH]
                for k, comp in enumerate(char_index):
                    data[i, j, k] = comp
                data_word[i, j] = word_index
    else:
        # do not use juman but use n-gram. not completed yet
        char_int_sequence = [text_to_char_index(full_vocab=full_vocab, real_vocab_number=real_vocab_number,
                                                chara_bukken_revised=chara_bukken_revised, sentence_text=text)
                             for text in sentence_texts]
        padded_char_sequence = pad_sequences(char_int_sequence, maxlen=MAX_WORD_LENGTH, )
        print("the n-gram module that not compeleted")
        exit(1)

    # convert labels to one-hot vectors
    labels_ng_c = to_categorical(numpy.asarray(labels_ng))
    labels_14_c = to_categorical(numpy.asarray(labels_14))
    print('Poly Label Shape:', labels_ng_c.shape)
    print('Fine-grained Label Shape:', labels_14_c.shape)

    # split data into training and validation
    indices = numpy.arange(data.shape[0])
    numpy.random.shuffle(indices)
    data = data[indices]
    data_word = data_word[indices]
    labels_ng_c = labels_ng_c[indices]
    labels_14_c = labels_14_c[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x1_train = data[:-nb_validation_samples]
    x2_train = data_word[:-nb_validation_samples]
    y1_train = labels_ng_c[:-nb_validation_samples]
    y2_train = labels_14_c[:-nb_validation_samples]
    x1_val = data[-nb_validation_samples:]
    x2_val = data_word[-nb_validation_samples:]
    y1_val = labels_ng_c[-nb_validation_samples:]
    y2_val = labels_14_c[-nb_validation_samples:]

    print('Number of different reviews for training and test')
    print(y1_train.sum(axis=0))
    print(y1_val.sum(axis=0))
    print(y2_train.sum(axis=0))
    print(y2_val.sum(axis=0))

    return full_vocab, real_vocab_number, chara_bukken_revised, word_vocab, \
           x1_train, x2_train, y1_train, y2_train, x1_val, x2_val, y1_val, y2_val


def do_kyoto_classification_task(dev_mode=False, juman=True, attention=False, char_emb_dim=CHAR_EMB_DIM, task="kyoto"):
    full_vocab, real_vocab_number, chara_bukken_revised, word_vocab, \
    x1_train, x2_train, y1_train, y2_train, x1_val, x2_val, y1_val, y2_val = \
        prepare_kyoto_classification(dev_mode=dev_mode, juman=juman)

    word_vocab_size = len(word_vocab)

    print("Do task: ", task)

    print("Char Only-2 classes")
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
    stopper = EarlyStopping(monitor='val_loss', patience=50)
    model1 = build_sentence_rnn(real_vocab_number=real_vocab_number, classes=2, attention=attention, word=False)
    model1.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['acc'])
    model1.fit(x1_train, y1_train, validation_data=(x1_val, y1_val),
               epochs=500, batch_size=BATCH_SIZE,
               callbacks=[reducelr, stopper])

    print("Char Only-14 classes")
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model2 = build_sentence_rnn(real_vocab_number=real_vocab_number, classes=14, attention=attention, word=False)
    model2.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['acc'])
    model2.fit(x1_train, y2_train, validation_data=(x1_val, y2_val),
               epochs=500, batch_size=BATCH_SIZE,
               callbacks=[reducelr, stopper])

    print("Word Only-2 classes")
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model3 = build_sentence_rnn(real_vocab_number=real_vocab_number, word_vocab_size=word_vocab_size,
                                classes=2, attention=attention, char_shape=False)
    model3.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['acc'])
    model3.fit(x2_train, y1_train, validation_data=(x2_val, y1_val),
               epochs=500, batch_size=BATCH_SIZE,
               callbacks=[reducelr, stopper])

    print("Word Only-14 classes")
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model4 = build_sentence_rnn(real_vocab_number=real_vocab_number, word_vocab_size=word_vocab_size,
                                classes=14, attention=attention, char_shape=False)
    model4.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['acc'])
    model4.fit(x2_train, y2_train, validation_data=(x2_val, y2_val),
               epochs=500, batch_size=BATCH_SIZE,
               callbacks=[reducelr, stopper])

    print("Word+Char-2 classes")
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model5 = build_sentence_rnn(real_vocab_number=real_vocab_number, word_vocab_size=word_vocab_size,
                                classes=2, attention=attention, word=True, char_shape=True)
    model5.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['acc'])
    model5.fit([x1_train, x2_train], y1_train, validation_data=([x1_val, x2_val], y1_val),
               epochs=500, batch_size=BATCH_SIZE,
               callbacks=[reducelr, stopper])

    print("Word+Char-14 classes")
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model6 = build_sentence_rnn(real_vocab_number=real_vocab_number, word_vocab_size=word_vocab_size,
                                classes=14, attention=attention, word=True, char_shape=True)
    model6.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['acc'])
    model6.fit([x1_train, x2_train], y2_train, validation_data=([x1_val, x2_val], y2_val),
               epochs=500, batch_size=BATCH_SIZE,
               callbacks=[reducelr, stopper])


def prepare_aozora_classification(dev_mode=False):
    # get vocab
    full_vocab, real_vocab_number, chara_bukken_revised, addtional_translate, _ = get_vocab()

    # if isinstance(sentence_input_text, str):
    #     sentence_input_text = sentence_input_text.split()
    # sentence_input_int = [text_to_char_index(full_vocab=full_vocab, real_vocab_number=real_vocab_number,
    #                                         chara_bukken_revised=chara_bukken_revised,
    #                                         sentence_text=x) for x in sentence_input_text]
    # preprocess data

    # read data. Shape: {"NDC 3,4,7": 1000 word sequence * 5000, "NDC 9": ...}
    data = pickle.load((open("aozora/aozora_big_5000_2lei.pickle", "rb")))

    label_names = list(data.keys())

    data_size = len(data[label_names[0]]) * len(label_names)

    global MAX_SENTENCE_LENGTH
    MAX_SENTENCE_LENGTH = 1000

    # change the sentence into matrix of word sequence
    data_char = numpy.zeros((data_size, MAX_SENTENCE_LENGTH, COMP_WIDTH * MAX_WORD_LENGTH), dtype=numpy.int32)
    data_word = numpy.zeros((data_size, MAX_SENTENCE_LENGTH), dtype=numpy.int32)
    print("Data shape: {shape}".format(shape=str(data_char.shape)))

    word_vocab = ["</s>"]
    labels = []

    for label, sentences in data.items():
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                if word not in word_vocab:
                    word_vocab.append(word)
                    word_index = len(word_vocab) - 1
                else:
                    word_index = word_vocab.index(word)
                char_index = text_to_char_index(full_vocab=full_vocab, real_vocab_number=real_vocab_number,
                                                chara_bukken_revised=chara_bukken_revised,
                                                addition_translate=addtional_translate, sentence_text=word)
                if len(char_index) < COMP_WIDTH * MAX_WORD_LENGTH:
                    char_index = char_index + [0] * (COMP_WIDTH * MAX_WORD_LENGTH - len(char_index))  # Padding
                elif len(char_index) > COMP_WIDTH * MAX_WORD_LENGTH:
                    char_index = char_index[:COMP_WIDTH * MAX_WORD_LENGTH]
                for k, comp in enumerate(char_index):
                    data_char[i, j, k] = comp
                data_word[i, j] = word_index
            labels.append(label_names.index(label))

    # convert labels to one-hot vectors
    labels = to_categorical(numpy.asarray(labels))
    print('Label Shape:', labels.shape)

    # split data into training and validation
    indices = numpy.arange(data_char.shape[0])
    numpy.random.shuffle(indices)
    data_char = data_char[indices]
    data_word = data_word[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data_char.shape[0])

    x1_train = data_char[:-nb_validation_samples]
    x2_train = data_word[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x1_val = data_char[-nb_validation_samples:]
    x2_val = data_word[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    print('Number of different reviews for training and test')
    print(y_train.sum(axis=0))
    print(y_val.sum(axis=0))

    with open("aozora_big_2_data.pickle", "wb") as f:
        pickle.dump((full_vocab, real_vocab_number, chara_bukken_revised, word_vocab,
                     x1_train, x2_train, y_train, x1_val, x2_val, y_val), f)

    return full_vocab, real_vocab_number, chara_bukken_revised, word_vocab, \
           x1_train, x2_train, y_train, x1_val, x2_val, y_val


def do_aozora_classification(dev_mode=False, attention=False, cnn_encoder=True):
    global MAX_SENTENCE_LENGTH
    MAX_SENTENCE_LENGTH = 1000

    with open("aozora_big_2_data.pickle", "rb") as f:
        (full_vocab, real_vocab_number, chara_bukken_revised, word_vocab,
         x1_train, x2_train, y_train, x1_val, x2_val, y_val) = pickle.load(f)
    word_vocab_size = len(word_vocab)

    num_class = 2

    print("Char Only")
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
    stopper = EarlyStopping(monitor='val_loss', patience=50)
    model2 = build_sentence_rnn(real_vocab_number=real_vocab_number, classes=num_class,
                                attention=attention, word=False, cnn_encoder=cnn_encoder)
    model2.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['acc'], )
    model2.fit(x1_train, y_train, validation_data=(x1_val, y_val),
               epochs=500, batch_size=BATCH_SIZE,
               callbacks=[reducelr, stopper])

    print("Word Only")
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model4 = build_sentence_rnn(real_vocab_number=real_vocab_number, word_vocab_size=word_vocab_size,
                                classes=num_class, attention=attention, char_shape=False,
                                cnn_encoder=cnn_encoder)
    model4.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['acc'])
    model4.fit(x2_train, y_train, validation_data=(x2_val, y_val),
               epochs=500, batch_size=BATCH_SIZE,
               callbacks=[reducelr, stopper])

    print("Word+Char-14 classes")
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model6 = build_sentence_rnn(real_vocab_number=real_vocab_number, word_vocab_size=word_vocab_size,
                                classes=num_class, attention=attention, word=True, char_shape=True,
                                cnn_encoder=cnn_encoder)
    model6.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['acc'])
    model6.fit([x1_train, x2_train], y_train, validation_data=([x1_val, x2_val], y_val),
               epochs=500, batch_size=BATCH_SIZE,
               callbacks=[reducelr, stopper])


def prepare_ChnSenti_classification(filename="ChnSentiCorp_htl_ba_6000/", dev_mode=False):
    # get vocab
    full_vocab, real_vocab_number, chara_bukken_revised, addtional_translate, _ = get_vocab()

    TEXT_DATA_DIR = filename
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    # maxlen = 0    # used the max length of sentence in the data. but large number makes cudnn die
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='gbk')
                try:
                    t = f.read()
                except UnicodeDecodeError:
                    continue
                t = t.translate(str.maketrans("", "", "\n"))
                t_list = list(jieba.cut(t, cut_all=False))
                # if len(t_list) > maxlen:
                #     maxlen = len(t_list)
                if len(t_list) > MAX_SENTENCE_LENGTH:
                    t_list = t_list[:MAX_SENTENCE_LENGTH]
                texts.append(t_list)
                f.close()
                labels.append(label_id)

    print('Found %s texts.' % len(texts))

    data_size = len(texts)

    # global MAX_SENTENCE_LENGTH
    # MAX_SENTENCE_LENGTH = maxlen

    # change the sentence into matrix of word sequence
    data_char = numpy.zeros((data_size, MAX_SENTENCE_LENGTH, COMP_WIDTH * MAX_WORD_LENGTH), dtype=numpy.int32)
    data_word = numpy.zeros((data_size, MAX_SENTENCE_LENGTH), dtype=numpy.int32)
    data_gram = numpy.zeros((data_size, MAX_SENTENCE_LENGTH, MAX_WORD_LENGTH), dtype=numpy.int32)
    print("Data shape: {shape}".format(shape=str(data_char.shape)))

    word_vocab = ["</s>"]
    gram_vocab = ["</s>"]

    for i, text in enumerate(texts):
        for j, word in enumerate(text):
            # word level
            if word not in word_vocab:
                word_vocab.append(word)
                word_index = len(word_vocab) - 1
            else:
                word_index = word_vocab.index(word)
            data_word[i, j] = word_index
            # single char gram level
            for l, char_g in enumerate(word):
                if char_g not in gram_vocab:
                    gram_vocab.append(char_g)
                    char_g_index = len(gram_vocab) - 1
                else:
                    char_g_index = gram_vocab.index(char_g)
                if l<MAX_WORD_LENGTH:
                    data_gram[i, j, l] = char_g_index
            # char shape level
            char_index = text_to_char_index(full_vocab=full_vocab, real_vocab_number=real_vocab_number,
                                            chara_bukken_revised=chara_bukken_revised,
                                            addition_translate=addtional_translate,
                                            sentence_text=word)
            if len(char_index) < COMP_WIDTH * MAX_WORD_LENGTH:
                char_index = char_index + [0] * (COMP_WIDTH * MAX_WORD_LENGTH - len(char_index))  # Padding
            elif len(char_index) > COMP_WIDTH * MAX_WORD_LENGTH:
                char_index = char_index[:COMP_WIDTH * MAX_WORD_LENGTH]
            for k, comp in enumerate(char_index):
                data_char[i, j, k] = comp

    # convert labels to one-hot vectors
    labels = to_categorical(numpy.asarray(labels))
    print('Label Shape:', labels.shape)

    # split data into training and validation
    indices = numpy.arange(data_char.shape[0])
    numpy.random.shuffle(indices)
    data_char = data_char[indices]
    data_word = data_word[indices]
    data_gram = data_gram[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data_char.shape[0])

    x1_train = data_char[:-nb_validation_samples]
    x2_train = data_word[:-nb_validation_samples]
    x3_train = data_gram[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x1_val = data_char[-nb_validation_samples:]
    x2_val = data_word[-nb_validation_samples:]
    x3_val = data_gram[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    print('Number of different reviews for training and test')
    print(y_train.sum(axis=0))
    print(y_val.sum(axis=0))

    return full_vocab, real_vocab_number, chara_bukken_revised, word_vocab, gram_vocab, \
           x1_train, x2_train, x3_train, y_train, x1_val, x2_val, x3_val, y_val


def do_ChnSenti_classification(filename, dev_mode=False, attention=False, cnn_encoder=True,
                               char_shape_only=True, char_only=True, word_only=True):
    (full_vocab, real_vocab_number, chara_bukken_revised, word_vocab, char_vocab,
     x1_train, x2_train, x3_train, y_train, x1_val, x2_val, x3_val, y_val) \
        = prepare_ChnSenti_classification(filename=filename, dev_mode=dev_mode)
    word_vocab_size = len(word_vocab)
    char_vocab_size = len(char_vocab)

    num_class = 2
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
    stopper = EarlyStopping(monitor='val_loss', patience=20)

    if char_shape_only:
        print("Char Shape Only")
        model2 = build_sentence_rnn(real_vocab_number=real_vocab_number, classes=num_class,
                                    attention=attention, word=False, char=False, cnn_encoder=cnn_encoder)
        model2.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop',
                       metrics=['acc'], )
        model2.fit(x1_train, y_train, validation_data=(x1_val, y_val),
                   epochs=100, batch_size=BATCH_SIZE,
                   callbacks=[reducelr, stopper])

    if char_only:
        print("Char Only")
        model2 = build_sentence_rnn(real_vocab_number=real_vocab_number, char_vocab_size=char_vocab_size,
                                    classes=num_class,
                                    attention=attention, word=False, char=True, char_shape=False, cnn_encoder=cnn_encoder)
        model2.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop',
                       metrics=['acc'], )
        model2.fit(x3_train, y_train, validation_data=(x3_val, y_val),
                   epochs=100, batch_size=BATCH_SIZE,
                   callbacks=[reducelr, stopper])

    if word_only:
        print("Word Only")
        model4 = build_sentence_rnn(real_vocab_number=real_vocab_number, word_vocab_size=word_vocab_size,
                                    classes=num_class, attention=attention, char_shape=False, char=False,
                                    cnn_encoder=cnn_encoder)
        model4.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop',
                       metrics=['acc'])
        model4.fit(x2_train, y_train, validation_data=(x2_val, y_val),
                   epochs=100, batch_size=BATCH_SIZE,
                   callbacks=[reducelr, stopper])

    # print("Word+Char")
    # sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    # model6 = build_sentence_rnn(real_vocab_number=real_vocab_number, word_vocab_size=word_vocab_size,
    #                             classes=num_class, attention=attention, word=True, char_shape=True,
    #                             cnn_encoder=cnn_encoder)
    # model6.compile(loss='categorical_crossentropy',
    #                optimizer=sgd,
    #                metrics=['acc'])
    # model6.fit([x1_train, x2_train], y_train, validation_data=([x1_val, x2_val], y_val),
    #            epochs=500, batch_size=BATCH_SIZE,
    #            callbacks=[reducelr, stopper])


def test_classifier(attention=False, cnn_encoder=True):
    x1_train_0 = numpy.random.normal(loc=4.0, scale=2.0, size=(500, MAX_SENTENCE_LENGTH, COMP_WIDTH * MAX_WORD_LENGTH))
    x1_train_1 = numpy.random.uniform(low=5, high=10, size=(500, MAX_SENTENCE_LENGTH, COMP_WIDTH * MAX_WORD_LENGTH))
    x2_train_0 = numpy.random.normal(loc=4.0, scale=2.0, size=(500, MAX_SENTENCE_LENGTH))
    x2_train_1 = numpy.random.uniform(low=5, high=10, size=(500, MAX_SENTENCE_LENGTH))
    x1_data = numpy.concatenate((x1_train_0, x1_train_1), axis=0)
    x2_data = numpy.concatenate((x2_train_0, x2_train_1), axis=0)
    labels = [0] * 500 + [1] * 500
    y_data = to_categorical(numpy.asarray(labels))

    indices = numpy.arange(x1_data.shape[0])
    numpy.random.shuffle(indices)
    data_char = x1_data[indices]
    data_word = x2_data[indices]
    y_data = y_data[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data_char.shape[0])

    x1_train = data_char[:-nb_validation_samples]
    x2_train = data_word[:-nb_validation_samples]
    y_train = y_data[:-nb_validation_samples]
    x1_val = data_char[-nb_validation_samples:]
    x2_val = data_word[-nb_validation_samples:]
    y_val = y_data[-nb_validation_samples:]

    word_vocab_size = 10

    print("Char Only")
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
    stopper = EarlyStopping(monitor='val_loss', patience=50)
    model2 = build_sentence_rnn(real_vocab_number=10, classes=2,
                                attention=attention, word=False, cnn_encoder=cnn_encoder)
    model2.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['acc'], )
    model2.fit(x1_train, y_train, validation_data=(x1_val, y_val),
               epochs=15, batch_size=BATCH_SIZE,
               callbacks=[reducelr, stopper])

    print("Word Only")
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model4 = build_sentence_rnn(real_vocab_number=10, word_vocab_size=word_vocab_size,
                                classes=2, attention=attention, char_shape=False,
                                cnn_encoder=cnn_encoder)
    model4.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['acc'])
    model4.fit(x2_train, y_train, validation_data=(x2_val, y_val),
               epochs=15, batch_size=BATCH_SIZE,
               callbacks=[reducelr, stopper])


if __name__ == "__main__":
    # Test Vocab
    # print(build_jp_embedding())
    #
    # for i in [4000, 5000, 8000]:
    #     print(full_vocab[i], chara_bukken_revised[i], [full_vocab[k] for k in chara_bukken_revised[i]])
    #
    # print(text_to_char_index(full_vocab=full_vocab, real_vocab_number=real_vocab_number,
    #                          chara_bukken_revised=chara_bukken_revised, mode="padding", comp_width=3))

    # from keras.models import Sequential

    # Test Pooling
    # model1 = Sequential()
    # model1.add(Embedding(input_dim=3, output_dim=6))
    # model1.add(AveragePooling1D(pool_size=3, strides=3))
    # model1.compile('rmsprop', 'mse')
    # input_array = numpy.random.randint(3, size=(30, 12))
    # output_array = model1.predict(input_array)
    # print(output_array.shape)

    # Test Word Encoder
    # model = build_word_feature()
    # model.compile('rmsprop', 'mse')
    # input_array = numpy.random.randint(5, size=(MAX_SENTENCE_LENGTH, COMP_WIDTH * MAX_WORD_LENGTH))
    # output_array = model.predict(input_array)
    # print(output_array.shape)
    # print(output_array[0])

    # Test Sentence Encoder
    # model = build_sentence_rnn(real_vocab_number=5, classes=2, attention=True, word=True, char_shape=True)
    # model.compile('rmsprop', 'mse')
    # input1_array = numpy.random.randint(5, size=(30, MAX_SENTENCE_LENGTH, COMP_WIDTH * MAX_WORD_LENGTH))
    # input2_array = numpy.random.randint(5, size=(30, MAX_SENTENCE_LENGTH, ))
    # output_array = model.predict([input1_array, input2_array])
    # print(output_array.shape)

    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png')

    # Test data preprocess
    # data = kyoto_classification_job(dev_mode=False, juman=True)
    # for words in data[120:129]:
    #     for word in words:
    #         for token in word:
    #             if token != 0:
    #                 print(full_vocab[token], end="")
    #     print("\n", end="")
    # print("no attention")
    # kyoto_classification()
    # print("attention")
    # do_kyoto_classification_task(attention=False, task="kyoto")
    # prepare_aozora_classification()
    # print("no cnn encoder")
    # do_aozora_classification(cnn_encoder=False)
    # print("use cnn encoder")
    # do_aozora_classification(cnn_encoder=True)
    # test_classifier()
    print("DATASET: CH2000")
    do_ChnSenti_classification(filename="ChnSentiCorp_htl_ba_2000/", char_shape_only=True, char_only=True, word_only=True)
    print("DATASET: CH4000")
    do_ChnSenti_classification(filename="ChnSentiCorp_htl_ba_4000/", char_shape_only=True, char_only=True, word_only=True)
    print("DATASET: CH6000")
    do_ChnSenti_classification(filename="ChnSentiCorp_htl_ba_6000/", char_shape_only=True, char_only=True, word_only=True)
