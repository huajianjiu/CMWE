import re, string, pickle, numpy, pandas, mojimoji, datetime
from pyknp import Jumanpp
from keras import optimizers
from keras.models import Model
from keras.layers import Embedding, Input, AveragePooling1D, MaxPooling1D, Conv1D, concatenate, TimeDistributed, \
    Bidirectional, LSTM, Dense, Flatten
from keras.legacy.layers import Highway
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.utils.np_utils import to_categorical
from getShapeCode import get_all_word_bukken
from keras.preprocessing.sequence import pad_sequences
from attention import AttentionWithContext

MAX_SENTENCE_LENGTH = 20
MAX_WORD_LENGTH = 3
COMP_WIDTH = 2
CHAR_EMB_DIM = 15
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 1000
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
    addition_translate = str.maketrans("ッャュョヮヵヶ", "っゃゅょゎゕゖ")

    hira_punc_number_latin = "".join(hirakana_list) + string.punctuation + \
                             ',)]｝、〕〉》」』】〙〗〟’”｠»ゝゞー' \
                             'ヴゎゕゖㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇷ゚ㇺㇻㇼㇽㇾㇿ々〻' \
                             '‐゠–〜～?!‼⁇⁈⁉・:;？、！/。.([｛〔〈《「『【〘〖〝‘“｟«—…‥〳〴〵･' \
                             '1234567890' \
                             'abcdefghijklmnopqrstuvwxyz ' \
                             '○●☆★■♪ヾω*´｀≧∇≦'
    # note: the space and punctuations in Jp sometimes show emotion

    vocab_chara, vocab_bukken, chara_bukken = get_all_word_bukken()
    hira_punc_number_latin_number = len(hira_punc_number_latin) + 2
    vocab = ["</padblank>", "</s>"] + list(hira_punc_number_latin) + vocab_bukken
    real_vocab_number = len(vocab)  # the part of the vocab that is really used. only basic components
    vocab_chara_strip = [chara for chara in vocab_chara if chara not in vocab_bukken]  # delete 独体字
    print("totally {n} puctuation, kana, and chara components".format(n=str(real_vocab_number)))
    full_vocab = vocab + vocab_chara_strip  # add unk at the head, and complex charas for text encoding at the tail
    chara_bukken_revised = {}
    for i_word, i_bukken in chara_bukken.items():  # update the index
        if vocab_chara[i_word] not in vocab_bukken:  # delete 独体字
            chara_bukken_revised[full_vocab.index(vocab_chara[i_word])] = \
                [k + hira_punc_number_latin_number for k in i_bukken]
    del vocab_chara
    del chara_bukken

    return full_vocab, real_vocab_number, chara_bukken_revised


def text_to_char_index(full_vocab, real_vocab_number, chara_bukken_revised, sentence_text,
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
    addition_translate = str.maketrans("ッャュョヮヵヶ￣▽", "っゃゅょゎゕゖ ∇")  # additional transformation
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
            i = ch2id[c]
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
            i = ch2id[c]
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


def build_word_feature(vocab_size=5, char_emb_dim=CHAR_EMB_DIM, comp_width=COMP_WIDTH,
                       mode="padding"):
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
    if mode == "average":
        # 2nd layer average the #comp_width components of every character
        char_embedding = AveragePooling1D(pool_size=comp_width, strides=comp_width, padding='valid')
        # TODO: conv, filter width 1 2 3, feature maps 50 100 150
        feature1 = Conv1D(filters=50, kernel_size=1)(char_embedding)
        feature1 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 1 + 1)(feature1)
        feature2 = Conv1D(filters=100, kernel_size=2)(char_embedding)
        feature2 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 2 + 1)(feature2)
        feature3 = Conv1D(filters=150, kernel_size=2)(char_embedding)
        feature3 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 3 + 1)(feature3)
        feature = concatenate([feature1, feature2])
    elif mode == "padding":
        # print(char_embedding._keras_shape)
        # TODO: conv, filter with [1, 2, 3]*#comp_width, feature maps 50 100 150
        feature1 = Conv1D(filters=50, kernel_size=1 * comp_width, strides=comp_width)(char_embedding)
        feature1 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 1 + 1)(feature1)
        feature2 = Conv1D(filters=100, kernel_size=2 * comp_width, strides=comp_width)(char_embedding)
        feature2 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 2 + 1)(feature2)
        feature3 = Conv1D(filters=150, kernel_size=3 * comp_width, strides=comp_width)(char_embedding)
        feature3 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 3 + 1)(feature3)
        feature = concatenate([feature1, feature2])
    feature = Flatten()(feature)
    # print(feature._keras_shape)
    feature = Highway()(feature)
    word_feature_encoder = Model(word_input, feature)
    return word_feature_encoder


def build_sentence_rnn(real_vocab_number, word_vocab_size=10, classes=2, attention=False, dropout=0, word=True, char=True):
    # build the rnn of words, use the output of build_word_feature as the feature of each word
    if char:
        word_feature_encoder = build_word_feature(vocab_size=real_vocab_number)
        sentence_input = Input(shape=(MAX_SENTENCE_LENGTH, COMP_WIDTH * MAX_WORD_LENGTH), dtype='int32')
        word_feature_sequence = TimeDistributed(word_feature_encoder)(sentence_input)
        # print(word_feature_sequence._keras_shape)
    if word:
        sentence_word_input = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
        word_embedding_sequence = Embedding(input_dim=word_vocab_size, output_dim=WORD_DIM)(sentence_word_input)
    if char and word:
        word_feature_sequence = concatenate([word_feature_sequence, word_embedding_sequence], axis=2)
        # print(word_feature_sequence._keras_shape)
    if word and not char:
        word_feature_sequence = word_embedding_sequence
    if attention:
        lstm_rnn = Bidirectional(LSTM(150, dropout=dropout, return_sequences=True))(word_feature_sequence)
        lstm_rnn = TimeDistributed(Highway())(lstm_rnn)
        lstm_rnn = AttentionWithContext()(lstm_rnn)
    else:
        lstm_rnn = Bidirectional(LSTM(150, dropout=dropout, return_sequences=False))(word_feature_sequence)
    # print(lstm_rnn._keras_shape)
    # lstm_rnn = TimeDistributed(Highway())(lstm_rnn)
    if classes < 2:
        print("class number cannot less than 2")
        exit(1)
    else:
        preds = Dense(classes, activation='softmax')(lstm_rnn)
    if char and not word:
        sentence_model = Model(sentence_input, preds)
    if word and not char:
        sentence_model = Model(sentence_word_input, preds)
    if word and char:
        sentence_model = Model([sentence_input, sentence_word_input], preds)
    sentence_model.summary()
    return sentence_model


def prepare_kyoto_classification(dev_mode=False, juman=True):
    # expected input. a sequence of the forward output of build_word_feature

    # get vocab
    full_vocab, real_vocab_number, chara_bukken_revised = get_vocab()

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
                                                chara_bukken_revised=chara_bukken_revised, sentence_text=word)
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

def prepare_aozora_classification(dev_mode=False):
    # get vocab
    full_vocab, real_vocab_number, chara_bukken_revised = get_vocab()

    # if isinstance(sentence_input_text, str):
    #     sentence_input_text = sentence_input_text.split()
    # sentence_input_int = [text_to_char_index(full_vocab=full_vocab, real_vocab_number=real_vocab_number,
    #                                         chara_bukken_revised=chara_bukken_revised,
    #                                         sentence_text=x) for x in sentence_input_text]
    # preprocess data

    # read xlsx
    data = pickle.load((open("/aozora/aozora.pickle", "rb")))

    label_names = list(data.keys())

    SEN_EACH = 4

    data_size = len(data[label_names[0]]) * SEN_EACH

    # change the sentence into matrix of word sequence
    data = numpy.zeros((data_size, MAX_SENTENCE_LENGTH, COMP_WIDTH * MAX_WORD_LENGTH), dtype=numpy.int32)
    data_word = numpy.zeros((data_size, MAX_SENTENCE_LENGTH), dtype=numpy.int32)
    print("Data shape: {shape}".format(shape=str(data.shape)))

    word_vocab = ["</s>"]

    # TODO: padding the sequences to SEN_EACH*MAX_SENTENCE_LENGTH for each document and expand to
    # TODO: [len(data)*SEN_EACH, MAX_SENTENCE_LENGTH]



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


def do_classification_task(dev_mode=False, juman=True, attention=False, char_emb_dim=CHAR_EMB_DIM, task="kyoto"):
    if task=="kyoto":
        full_vocab, real_vocab_number, chara_bukken_revised, word_vocab, \
        x1_train, x2_train, y1_train, y2_train, x1_val, x2_val, y1_val, y2_val = \
            prepare_kyoto_classification(dev_mode=dev_mode, juman=juman, char_emb_dim=char_emb_dim)
    else:
        full_vocab, real_vocab_number, chara_bukken_revised, word_vocab, \
        x1_train, x2_train, y1_train, y2_train, x1_val, x2_val, y1_val, y2_val = \
            prepare_kyoto_classification(dev_mode=dev_mode, juman=juman, char_emb_dim=char_emb_dim)

    word_vocab_size = len(word_vocab)

    print("Do task: ", task)

    print("Char Only-2 classes")
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model1 = build_sentence_rnn(real_vocab_number=real_vocab_number, classes=2, attention=attention, word=False)
    model1.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['acc'])
    model1.fit(x1_train, y1_train, validation_data=(x1_val, y1_val),
               epochs=5000, batch_size=BATCH_SIZE,
               )

    print("Char Only-14 classes")
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model2 = build_sentence_rnn(real_vocab_number=real_vocab_number, classes=14, attention=attention, word=False)
    model2.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['acc'])
    model2.fit(x1_train, y2_train, validation_data=(x1_val, y2_val),
               epochs=5000, batch_size=BATCH_SIZE,
               )

    print("Word Only-2 classes")
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model3 = build_sentence_rnn(real_vocab_number=real_vocab_number, word_vocab_size=word_vocab_size,
                                classes=2, attention=attention, char=False)
    model3.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['acc'])
    model3.fit(x2_train, y1_train, validation_data=(x2_val, y1_val),
               epochs=500, batch_size=BATCH_SIZE,
               )

    print("Word Only-14 classes")
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model4 = build_sentence_rnn(real_vocab_number=real_vocab_number, word_vocab_size=word_vocab_size,
                                classes=14, attention=attention, char=False)
    model4.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['acc'])
    model4.fit(x2_train, y2_train, validation_data=(x2_val, y2_val),
               epochs=500, batch_size=BATCH_SIZE,
               )

    print("Word+Char-2 classes")
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model5 = build_sentence_rnn(real_vocab_number=real_vocab_number, word_vocab_size=word_vocab_size,
                                classes=2, attention=attention, word=True, char=True)
    model5.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['acc'])
    model5.fit([x1_train, x2_train], y1_train, validation_data=([x1_val, x2_val], y1_val),
               epochs=5000, batch_size=BATCH_SIZE,
               )

    print("Word+Char-14 classes")
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model6 = build_sentence_rnn(real_vocab_number=real_vocab_number, word_vocab_size=word_vocab_size,
                                classes=14, attention=attention, word=True, char=True)
    model6.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['acc'])
    model6.fit([x1_train, x2_train], y2_train, validation_data=([x1_val, x2_val], y2_val),
               epochs=5000, batch_size=BATCH_SIZE,
               )

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
    # model = build_sentence_rnn(real_vocab_number=5, classes=2, attention=True, word=True, char=True)
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
    do_classification_task(attention=False, task="kyoto")
