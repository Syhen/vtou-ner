# -*- coding: utf-8 -*-
"""
create on 2018-12-27 下午4:42

author @heyao
"""
import json
import re

from vtou_ner.utils.preprocessing import text2sentences

PATTERN_DICT = re.compile(r"(.*?)\{\{(.*?)[::](.*?)\}\}")


def extract_tags(line: str, meta_tags: str = "BMEO"):
    """transform {{key: sentence}} lines to
    :param line: str. 
    :param meta_tags: str. begin, middle, end, ordinal tags
    :return: 
    """
    t_begin, t_middle, t_end, t_ordinal = list(meta_tags)

    texts = []
    tags = []
    all_finds = PATTERN_DICT.findall(line)
    tags_len = 0
    for ordinal, tag, entity in all_finds:
        tags_len += (len(tag) + 5)
        # 实体前的文本
        texts.extend(list(ordinal))
        tags.extend([t_ordinal] * len(ordinal))
        # 实体
        texts.extend(list(entity))
        tags.append('%s_%s' % (t_begin, tag))
        tags.extend(['%s_%s' % (t_middle, tag) for _ in range(len(entity) - 2)])
        if len(entity) > 1:
            tags.append('%s_%s' % (t_end, tag))
    if all_finds:
        entity = all_finds[-1][2]
        remain_text = line[line.find(entity, len(texts) + tags_len - len(entity) - 2) + len(entity) + 2:]
        texts.extend(list(remain_text))
        tags.extend([t_ordinal] * len(remain_text))
    else:
        texts.extend(list(line))
        tags.extend([t_ordinal] * len(line))
    return texts, tags


def extract_boson_data(filename, meta_tags="BMEO", split=True):
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not split:
                texts, tags = extract_tags(line, meta_tags=meta_tags)
                yield texts, tags
                continue
            for line in text2sentences(line):
                texts, tags = extract_tags(line, meta_tags=meta_tags)
                yield texts, tags


def write_boson_data(from_filename, to_filename, meta_tags="BMEO", fmt="{0}/{1} ", split=True):
    data = extract_boson_data(from_filename, meta_tags=meta_tags, split=split)
    if fmt == 'json':
        if not to_filename.endswith('.json'):
            raise ValueError("fmt json, to_filename's extension must be .json")
        o = [{'texts': texts, 'tags': tags} for texts, tags in data]
        with open(to_filename, 'w') as f:
            json.dump(o, f, ensure_ascii=False)
        return True
    to_texts = '\n'.join(''.join(fmt.format(*i) for i in zip(texts, tags)) for texts, tags in data)
    with open(to_filename, 'w') as f:
        f.write(to_texts)


if __name__ == '__main__':
    # text = "[太开心]{{product_name:开心购物网}}推荐[围观]【{{company_name:idfix旗舰店}}】主营新品, t恤,半身裙,id.fix, idf, ol,长袖,{{time:春}}装,纯色,翻领,蕾丝,女式连衣裙,{{time:秋}}{{time:冬}}装,{{time:秋}}装,时尚,外套,无袖,{{time:夏}}装,新款共有宝贝327件，近期累计成交3409笔，专柜品质，值得信赖！[给力]==>{{product_name:http://t.cn/8s94E5J}}"
    # texts, tags = extract_tags(text)
    # print(list(zip(texts, tags)))
    write_boson_data("origindata.txt", "fmt_boson_data_split.json", fmt="json", split=True)
