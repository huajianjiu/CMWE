import re, string, pickle, numpy, pandas
from keras.models import Model
from keras.layers import Embedding, Input, AveragePooling1D, MaxPooling1D, Conv1D, concatenate, TimeDistributed, \
    Bidirectional, LSTM, Dense, Flatten
from keras.legacy.layers import Highway
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.engine import Layer
from getShapeCode import get_all_word_bukken

MAX_SENTENCE_LENGTH = 100
MAX_WORD_LENGTH = 7
COMP_WIDTH = 3
CHAR_EMB_DIM = 15


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

    hira_and_punc = "".join(hirakana_list) + string.punctuation + \
                    ',)]｝、〕〉》」』】〙〗〟’”｠»ゝゞー' \
                    'ヴゎゕゖㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇷ゚ㇺㇻㇼㇽㇾㇿ々〻' \
                    '‐゠–〜～?!‼⁇⁈⁉・:;？、！/。.([｛〔〈《「『【〘〖〝‘“｟«—…‥〳〴〵'

    vocab_chara, vocab_bukken, chara_bukken = get_all_word_bukken()
    hira_and_punc_number = len(hira_and_punc) + 2
    vocab = ["</padblank>", "</s>"] + list(hira_and_punc) + vocab_bukken
    real_vocab_number = len(vocab)  # the part of the vocab that is really used. only basic components
    vocab_chara = [chara for chara in vocab_chara if chara not in vocab_bukken]  # delete 独体字
    print("totally {n} puctuation, kana, and chara components".format(n=str(real_vocab_number)))
    full_vocab = vocab + vocab_chara  # add unk at the head, and complex charas for text encoding at the tail
    chara_bukken_revised = {}
    for i_word, i_bukken in chara_bukken.items():  # update the index
        chara_bukken_revised[i_word + real_vocab_number] = \
            [k + hira_and_punc_number for k in i_bukken]
    del chara_bukken

    return full_vocab, real_vocab_number, chara_bukken_revised


def text_to_char_index(full_vocab=[], real_vocab_number=0, chara_bukken_revised={}, sentence_text="今日は何シテ遊ぶの？",
                       mode="padding", char_emb_dim=CHAR_EMB_DIM, comp_width=COMP_WIDTH):
    # mode:
    # average: will repeat the original index to #comp_width for the process of the embedding layer
    # padding: will pad the original index to #comp_width with zero for the process of the embedding layer
    # char_emb_dim  char embedding size
    # comp_width  #components used

    # convert kata to hira
    _, katakana2hiragana, _ = _make_kana_convertor()
    text = katakana2hiragana(sentence_text)
    addition_translate = str.maketrans("ッャュョヮヵヶ", "っゃゅょゎゕゖ")
    text = text.translate(addition_translate)
    # expanding every character with 3 components
    ch2id = {}
    for i, w in enumerate(full_vocab):
        ch2id[w] = i
    int_text = []
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
    word_input = Input(shape=(3*MAX_WORD_LENGTH,))
    char_embedding = \
        Embedding(input_dim=vocab_size, output_dim=char_emb_dim, weights=[init_weight], trainable=True)(word_input)
    if mode == "average":
        # 2nd layer average the #comp_width components of every character
        char_embedding = AveragePooling1D(pool_size=comp_width, strides=comp_width, padding='valid')
        # TODO: conv, filter width 1 2 3, feature maps 200 200 200
        feature1 = Conv1D(filters=200, kernel_size=1)(char_embedding)
        feature1 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 1 + 1)(feature1)
        feature2 = Conv1D(filters=200, kernel_size=2)(char_embedding)
        feature2 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 2 + 1)(feature2)
        feature3 = Conv1D(filters=200, kernel_size=2)(char_embedding)
        feature3 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 3 + 1)(feature3)
        feature = concatenate([feature1, feature2])
    elif mode == "padding":
        # print(char_embedding._keras_shape)
        # TODO: conv, filter with 1*#comp_width 2*#comp_width, feature maps 200 200
        feature1 = Conv1D(filters=200, kernel_size=1 * comp_width, strides=comp_width)(char_embedding)
        feature1 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 1 + 1)(feature1)
        feature2 = Conv1D(filters=200, kernel_size=2 * comp_width, strides=comp_width)(char_embedding)
        feature2 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 2 + 1)(feature2)
        feature3 = Conv1D(filters=200, kernel_size=3 * comp_width, strides=comp_width)(char_embedding)
        feature3 = MaxPooling1D(pool_size=MAX_WORD_LENGTH - 3 + 1)(feature3)
        feature = concatenate([feature1, feature2])
    feature = Flatten()(feature)
    # print(feature._keras_shape)
    feature = Highway()(feature)
    word_feature_encoder = Model(word_input, feature)
    return word_feature_encoder


def build_sentence_rnn(real_vocab_number, classes=2):
    # build the rnn of words, use the output of build_word_feature as the feature of each word
    sentence_input = Input(shape=(MAX_SENTENCE_LENGTH, 3*MAX_WORD_LENGTH))
    word_feature_encoder = build_word_feature(vocab_size=real_vocab_number)
    word_feature_sequence = TimeDistributed(word_feature_encoder)(sentence_input)
    lstm_rnn = Bidirectional(LSTM(300, dropout=0.5, return_sequences=True))(word_feature_sequence)
    lstm_rnn = TimeDistributed(Highway())(lstm_rnn)
    if classes == 2:
        preds = Dense(1, activation='sigmoid')(lstm_rnn)
    else:
        preds = Dense(classes, activation='softmax')(lstm_rnn)
    sentence_model = Model(sentence_input, preds)
    sentence_model.summary()
    return sentence_model

def kyoto_classification_job(dev_mode=False, char_emb_dim=CHAR_EMB_DIM):
    # expected input. a sequence of the forward output of build_word_feature

    # get vocab
    full_vocab, real_vocab_number, chara_bukken_revised = get_vocab()

    # if isinstance(sentence_input_text, str):
    #     sentence_input_text = sentence_input_text.split()
    # sentence_input_int = [text_to_char_index(full_vocab=full_vocab, real_vocab_number=real_vocab_number,
    #                                         chara_bukken_revised=chara_bukken_revised,
    #                                         sentence_text=x) for x in sentence_input_text]
    # TODO: preprocess data

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

    # TODO: change the sentence into word sequence

if __name__ == "__main__":
    # Test Vocab
    # print(build_jp_embedding())
    # full_vocab, real_vocab_number, chara_bukken_revised = get_vocab()
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
    # inputs = Input(shape=(3 * MAX_SENTENCE_LENGTH,))
    # outputs = build_word_feature(word_input=inputs)
    # model = Model(inputs=inputs, outputs=outputs)
    # model.compile('rmsprop', 'mse')
    # input_array = numpy.random.randint(5, size=(30, 3 * MAX_WORD_LENGTH))
    # output_array = model.predict(input_array)
    # print(output_array.shape)
    # print(output_array[0])

    # Test Sentence Encoder
    model = build_sentence_rnn(5)
    model.compile('rmsprop', 'mse')
    input_array = numpy.random.randint(5, size=(30, MAX_SENTENCE_LENGTH, 3 * MAX_WORD_LENGTH))
    output_array = model.predict(input_array)
    print(output_array.shape)
