# Transformers (Code)



## 加载数据集

```bash
# huggingface的数据集库
pip install datasets
```

```py
from datasets import load_dataset

ner_datasets = load_dataset("peoples_daily_ner",cache_dir="./data",trust_remote_code=True)
```



### 其他操作

#### 查看数据集格式

```py
ner_datasets
```

```bash
DatasetDict({
    train: Dataset({
        features: ['id', 'tokens', 'ner_tags'],
        num_rows: 20865
    })
    validation: Dataset({
        features: ['id', 'tokens', 'ner_tags'],
        num_rows: 2319
    })
    test: Dataset({
        features: ['id', 'tokens', 'ner_tags'],
        num_rows: 4637
    })
})
```

类型为`DatasetDict`

数据集分为：`train`（训练集）、`validation`（验证集）、`test`（测试集）

每个集合内的数据列名`feature`，和数据条数`num_rows`



#### 查看训练集的第一条数据内容

```py
ner_datasets["train"][0]
```

```json
{
    'id': '0',
    'tokens':['海','钓','比','赛','地','点','在','厦','门','与','金','门','之','间','的','海','域','。'],
    'ner_tags': [0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 5, 6, 0, 0, 0, 0, 0, 0]
}
```



#### 查看数据集属性（feature）

```py
ner_datasets["train"].features
```

```json
{
    'id': Value(dtype='string', id=None),
    'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
    'ner_tags': Sequence(feature=ClassLabel(names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'], id=None), length=-1, id=None)}
```



## 数据处理

```python
from trans
```











## 语音处理

## 修改采样率

```py
minds = load_dataset(
    "PolyAI/minds14",
    name="en-AU",
    split="train",
    trust_remote_code=True,
    cache_dir="./data",
)
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
```

```py
import librosa
import soundfile as sf

def resample_audio_librosa(input_path, output_path, target_sr=16000):
    # 使用 librosa 加载音频文件
    speech_array, orig_sr = librosa.load(input_path, sr=None)
    
    # 重采样到目标采样率
    if orig_sr != target_sr:
        speech_array = librosa.resample(speech_array, orig_sr=orig_sr, target_sr=target_sr)
    
    # 将重采样后的音频保存到输出文件
    sf.write(output_path, speech_array, target_sr)
```



## 绘图

波形图 waveform

```py
import matplotlib.pyplot as plt
import librosa

data, samplerate = librosa.load("../test.wav", sr=16000)

# 创建一个4行1列个子图的图表 (axes:轴或子图)
# 图标尺寸为18宽15高
# 每个子图的高度比例，前3个高度相同，第四个高度是前3个的1.75倍
axs = plt.subplots(4,1, figsize=(18,15), gridspec_kw={'height_ratios': [1, 1, 1, 1.75]})

# 第一个子图设置标题
axs[0].set_title("Waveform")
# 设置y轴标签
axs[0].set_ylabel('Amplitude')
axs[0].set_xlabel('Time (s)')

librosa.display.waveshow(data, sr=samplerate, ax=axs[0])
```



声谱图 spectrogram

