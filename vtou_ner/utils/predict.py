# -*- coding: utf-8 -*-
"""
create on 2018-12-27 下午9:33

author @heyao
"""


def tag_ids2entities(tag_ids, id2tag_mapping, meta_tags="BMEO"):
    """将tagid转换成entity列表
    :param tag_ids: list. id列表
    :param id2tag_mapping: dict. id到tag的转换表
    :param meta_tags: str. start, middle, end, ordinal tag.
    :return: "%s_%s_%s" % (start_index, entity_category, entity_len) 
    """
    t_start, t_middle, t_end, t_ordinal = list(meta_tags)
    entities = []
    start_index = 0  # entity开始说声音
    entity_len = 0  # entity长度
    entity_category = ""  # entity类型
    has_entity = False  # 是否有entity
    for i, tag_id in enumerate(tag_ids):
        if tag_id == 0:  # 如果是padding符号，直接跳过
            continue
        if id2tag_mapping[tag_id] == t_ordinal:  # 如果是常规字符，则判断有无实体（解决单独的B）
            if not has_entity:
                continue
            entity_len = i - start_index + 1
            entities.append("%s_%s_%s" % (start_index, entity_category, entity_len))
            has_entity = False
            continue
        status, tag = id2tag_mapping[tag_id].split('_', 1)
        if status == t_start:
            entity_category = tag
            start_index = i
            has_entity = True
        if status == t_end:
            entity_len = i - start_index + 1
            entities.append("%s_%s_%s" % (start_index, entity_category, entity_len))
            has_entity = False
    return entities


if __name__ == '__main__':
    tag2id_mapping = {
        'O': 1, 'M_product_name': 2, 'M_time': 3, 'M_org_name': 4, 'M_person_name': 5,
        'M_company_name': 6, 'B_person_name': 7, 'E_person_name': 8, 'B_location': 9, 'M_location': 10,
        'B_time': 11, 'E_time': 12, 'E_location': 13, 'B_product_name': 14, 'E_product_name': 15,
        'B_org_name': 16, 'E_org_name': 17, 'B_company_name': 18, 'E_company_name': 19
    }
    id2tag_mapping = dict(zip(tag2id_mapping.values(), tag2id_mapping.keys()))
    answer = ['3_person_name_2', '6_company_name_4']
    assert tag_ids2entities([1, 1, 1, 7, 8, 1, 18, 6, 6, 19, 0, 0],
                            id2tag_mapping) == answer, "some error to transform entity"
