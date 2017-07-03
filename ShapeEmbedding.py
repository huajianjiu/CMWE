import re, string, pickle
from keras.layers import Embedding
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.engine import Layer
from getShapeCode import get_all_word_bukken


class ChCharaEmbedding(Layer):
    def __init__(self, input_dim, output_dim, chara_bukken,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 **kwargs):
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        super(ChCharaEmbedding, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.input_length = input_length
        self.chara_bukken = chara_bukken

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name='embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            dtype=self.dtype)
        self.built = True

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(inputs, 0)

    def compute_output_shape(self, input_shape):
        if self.input_length is None:
            return input_shape + (self.output_dim,)
        else:
            # input_length can be tuple if input is 3D or higher
            if isinstance(self.input_length, (list, tuple)):
                in_lens = list(self.input_length)
            else:
                in_lens = [self.input_length]
            if len(in_lens) != len(input_shape) - 1:
                ValueError('"input_length" is %s, but received input has shape %s' %
                           (str(self.input_length), str(input_shape)))
            else:
                for i, (s1, s2) in enumerate(zip(in_lens, input_shape[1:])):
                    if s1 is not None and s2 is not None and s1 != s2:
                        ValueError('"input_length" is %s, but received input has shape %s' %
                                   (str(self.input_length), str(input_shape)))
                    elif s1 is None:
                        in_lens[i] = s2
            return (input_shape[0],) + tuple(in_lens) + (self.output_dim,)

    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        if self.chara_bukken is not None:
            for input in inputs:
                if input > self.input_dim:
                    pass
        # TODO: for the basic charas, return their embeddings;
        # TODO: for the complex charas, return the avr of their component embeddings.
        # TODO: discriminate them by their index. the
        out = K.gather(self.embeddings, inputs)
        return out

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'output_dim': self.output_dim,
                  'embeddings_initializer': initializers.serialize(self.embeddings_initializer),
                  'embeddings_regularizer': regularizers.serialize(self.embeddings_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'embeddings_constraint': constraints.serialize(self.embeddings_constraint),
                  'mask_zero': self.mask_zero,
                  'input_length': self.input_length}
        base_config = super(ChCharaEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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


def build_jp_embedding(opts=None):
    # convert kata to hira
    char_emb_dim = 15
    use_component = True # True for component level False for chara level

    _, _, hirakana_list = _make_kana_convertor()
    addition_translate = str.maketrans("ッャュョヮヵヶ", "っゃゅょゎゕゖ")

    hira_and_punc = "".join(hirakana_list) + string.punctuation + \
                    ',)]｝、〕〉》」』】〙〗〟’”｠»ゝゞー' \
                    'ヴゎゕゖㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇷ゚ㇺㇻㇼㇽㇾㇿ々〻' \
                    '‐゠–〜～?!‼⁇⁈⁉・:;/。.([｛〔〈《「『【〘〖〝‘“｟«—…‥〳〴〵'

    vocab_chara, vocab_bukken, chara_bukken = get_all_word_bukken()
    hira_and_punc_number = len(hira_and_punc) + 1
    vocab = ["</s>"] + list(hira_and_punc) + vocab_bukken
    real_vocab_number = len(vocab)  # the part of the vocab that is really used. only basic components
    print ("totally {n} puctuation, kana, and chara components".format(n=str(real_vocab_number)))
    full_vocab = vocab + vocab_chara  # add unk at the head, and complex charas for text encoding at the tail
    chara_bukken_revised = {}
    for i_word, i_bukken in chara_bukken.items():  # update the index
        chara_bukken_revised[i_word + real_vocab_number] = \
            [k+hira_and_punc_number for k in i_bukken]
    del chara_bukken

    # build embedding layer
    if use_component:
        char_embedding = ChCharaEmbedding(input_dim=real_vocab_number, output_dim=char_emb_dim, chara_bukken=chara_bukken_revised)
    else:
        char_embedding = Embedding(input_dim=len(full_vocab), output_dim=15)

    return full_vocab, real_vocab_number, chara_bukken_revised, char_embedding


def sentence_preprocess(sentence_text="今日は何シテ遊ぶの？"):
    # convert kata to hira
    char_emb_dim = 15
    _, katakana2hiragana, _ = _make_kana_convertor()
    text = katakana2hiragana(sentence_text)
    addition_translate = str.maketrans("ッャュョヮヵヶ", "っゃゅょゎゕゖ")
    text = text.translate(addition_translate)
    return text

if __name__ == "__main__":
    # print(build_jp_embedding())
    full_vocab, real_vocab_number, chara_bukken_revised, _ = build_jp_embedding()

    for i in [4000, 5000, 8000]:
        print(full_vocab[i], [full_vocab[k] for k in chara_bukken_revised[i]])
