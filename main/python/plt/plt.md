

# [python教程] PLT

```py
# 创建一个4行1列个子图的图表
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

