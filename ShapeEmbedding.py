import re, string, pickle, numpy
from keras.layers import Embedding
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.engine import Layer
from getShapeCode import get_all_word_bukken


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
    char_emb_dim = 15
    use_component = True # True for component level False for chara level

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
    print ("totally {n} puctuation, kana, and chara components".format(n=str(real_vocab_number)))
    full_vocab = vocab + vocab_chara  # add unk at the head, and complex charas for text encoding at the tail
    chara_bukken_revised = {}
    for i_word, i_bukken in chara_bukken.items():  # update the index
        chara_bukken_revised[i_word + real_vocab_number] = \
            [k+hira_and_punc_number for k in i_bukken]
    del chara_bukken

    return full_vocab, real_vocab_number, chara_bukken_revised


def sentence_preprocess(full_vocab=[], real_vocab_number=0, chara_bukken_revised={}, sentence_text="今日は何シテ遊ぶの？"):
    # convert kata to hira
    char_emb_dim = 15
    comp_width = 2  # the number of components used
    _, katakana2hiragana, _ = _make_kana_convertor()
    text = katakana2hiragana(sentence_text)
    addition_translate = str.maketrans("ッャュョヮヵヶ", "っゃゅょゎゕゖ")
    text = text.translate(addition_translate)
    # expanding every character with 3 components
    ch2id={}
    for i, w in enumerate(full_vocab):
        ch2id[w] = i
    int_text = []
    for c in text:
        i = ch2id[c]
        if i > real_vocab_number:
            comps = chara_bukken_revised[i]
            if len(comps) >= comp_width:
                int_text += comps[:comp_width]
            elif len(comps) == 1:
                int_text += [i]*comp_width
            else:
                int_text += comps + [0]*(comp_width - len(comps))
        else:
            int_text += [i]*comp_width
    return int_text


def build_char_embedding(vocab_size=5, char_emb_dim=15):
    # real vocab_size for ucs is 2481, including paddingblank, unkown, puncutations, kanas
    init_width = 0.5 / char_emb_dim
    init_weight = numpy.random.uniform(low=-init_width, high=init_width, size=[vocab_size, char_emb_dim])
    # init_weight[0] = 0  # maybe the padding should not be zero
    print(init_weight)
    # expected input: every 3 int express a character.
    # expected out: average of the 3 indexed embeddings of a character
    # first layer embeds every components
    char_embedding = Embedding(input_dim=vocab_size, output_dim=char_emb_dim, weights=init_weight)
    # TODO: 2nd layer average the 3 components of every character

if __name__ == "__main__":
    # print(build_jp_embedding())
    full_vocab, real_vocab_number, chara_bukken_revised = get_vocab()

    for i in [4000, 5000, 8000]:
        print(full_vocab[i], chara_bukken_revised[i], [full_vocab[k] for k in chara_bukken_revised[i]])

    print(sentence_preprocess(full_vocab=full_vocab, real_vocab_number=real_vocab_number,
                              chara_bukken_revised=chara_bukken_revised))