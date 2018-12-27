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

# next work
1. 微调参数，使得模型在验证集上表现良好
2. 使用[bert](https://github.com/huggingface/pytorch-pretrained-BERT)
