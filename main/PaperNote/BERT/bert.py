# AutoTokenizer会根据你的内容选择不同的分词器
from transformers import AutoTokenizer

# 从HuggingFace加载，输入模型名称，即可加载对于的分词器
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")

# 也可以将tokenizer 保存到本地（默认在 C:\Users\田合兰\.cache\huggingface\hub\models--uer--roberta-base-finetuned-dianping-chinese\snapshots\25faf1874b21e76db31ea9c396ccf2a0322e0071）
tokenizer.save_pretrained("./roberta_tokenizer")
# 从本地加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("./roberta_tokenizer/")

sen = "弱小的我也有大梦想!"
tokens = tokenizer.tokenize(sen)
# ['弱', '小', '的', '我', '也', '有', '大', '梦', '想', '!']

# 查看词典




