# coding:utf8
import torch as t
import numpy as np
import json
import jieba
import tqdm


class Config:
    annotation_file = 'intestines_image_caption_train_json.json'
    unknown = '</UNKNOWN>'
    end = '</EOS>'
    padding = '</PAD>'
    max_words = 2000
    min_appear = 1
    save_path = 'caption.pth'


def process(**kwargs):
    opt = Config()
    for k, v in kwargs.items():
        setattr(opt, k, v)

    with open(opt.annotation_file, encoding='utf-8') as f:
        data = json.load(f)


    id2ix = {item['image_id']: ix for ix, item in enumerate(data)}
    ix2id = {ix: id for id, ix in (id2ix.items())}
    assert id2ix[ix2id[10]] == 10

    captions = [item['caption'] for item in data]

    cut_captions = [[list(jieba.cut(ii, cut_all=False)) for ii in item] for item in tqdm.tqdm(captions)]

    word_nums = {}

    def update(word_nums):
        def fun(word):
            word_nums[word] = word_nums.get(word, 0) + 1
            return None

        return fun

    lambda_ = update(word_nums)
    _ = {lambda_(word) for sentences in cut_captions for sentence in sentences for word in sentence}


    word_nums_list = sorted([(num, word) for word, num in word_nums.items()], reverse=True)

    words = [word[1] for word in word_nums_list[:opt.max_words] if word[0] >= opt.min_appear]
    words = [opt.unknown, opt.padding, opt.end] + words
    word2ix = {word: ix for ix, word in enumerate(words)}
    ix2word = {ix: word for word, ix in word2ix.items()}
    assert word2ix[ix2word[123]] == 123

    ix_captions = [[[word2ix.get(word, word2ix.get(opt.unknown)) for word in sentence]
                    for sentence in item]
                   for item in cut_captions]

    results = {
        'caption': ix_captions,
        'word2ix': word2ix,
        'ix2word': ix2word,
        'ix2id': ix2id,
        'id2ix': id2ix,
        'padding': '</PAD>',
        'end': '</EOS>'
    }
    t.save(results, opt.save_path)
    print('save file in %s' % opt.save_path)

    def test(ix, ix2=4):
        results = t.load(opt.save_path)
        ix2word = results['ix2word']
        print(ix2word)
        print(results['caption'][ix])
        print(len(results['caption'][ix]))
        examples = results['caption'][ix][4]
        sentences_p = (''.join([ix2word[ii] for ii in examples]))
        sentences_r = data[ix]['caption'][ix2]
        assert sentences_p == sentences_r, 'test failed'

    test(1)
    print('test success')


if __name__ == '__main__':
    import fire

    fire.Fire()
