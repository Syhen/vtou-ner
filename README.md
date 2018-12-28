# vtou-ner
财经新闻命名实体识别

# Description
经典的Bi-LSTM-CRF模型. 其中CRF层使用的是：[这里](https://github.com/kolloldas/torchnlp/blob/master/torchnlp/modules/crf.py)

由于没有合适的财经新闻语料库，故使用[boson](https://bosonnlp.com/dev/resource)
命名实体识别语料库代替。后期新增了财经新闻命名实体识别语料库后，可直接替换。

# requirements
python==3.5

pytorch==1.0

numpy

pandas

# running
```commandline
# 安装依赖
cd vtou-ner
pip install -r requirements.txt
# 预处理数据
cd datasets
python extractor.py

cd ../bin
python train.py
```

# metrics
使用F1评估模型的表现

F1 = 2 * recall * precision / (recall + precision)

其中，recall = tp / (tp + fn), precision = tp / (tp + fp)

tp定义为 len(tag_true & tag_pred)

fp定义为 len(tag_pred - tag_true)

fn定义为 len(tag_true - tag_pred)

对于tag_true或tag_pred，模型的输出是：[1, 1, 1, 7, 8, 1, 18, 6, 6, 19, 0, 0]，
转换为：['3_person_name_2', '6_company_name_4']

依次表示为：实体开始索引，实体类别，实体长度，保证了实体提取的位置、类型、边界的一致

其他项目中[boson数据集的 f1 score 是70%~75%](https://github.com/buppt/ChineseNER)

本项目：
- embedding dim: 100
- fc hidden dim: 200
- lstm dim: 100, bidirectional: True
- max sentence len: 60
- epochs: 9
- dropout prob: 0.5

将boson数据由段，按照句号、叹号等分割成句子，对每句话进行实体标注
- vocab: 3916
- tag size: 21
- train on: 9800 samples
- validate on: 1225 samples
- test on: 1225 samples

f1 on validate set: 0.7113

f1 on test set: 0.7239

# next work
1. 微调参数，使得模型在验证集上表现良好
2. 使用[bert](https://github.com/huggingface/pytorch-pretrained-BERT)
