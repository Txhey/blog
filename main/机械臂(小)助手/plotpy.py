import plotly.graph_objects as go
import numpy as np

# 生成一些示例数据
x = np.linspace(0, 10, 1000)
y = np.sin(x)

# 创建 Plotly 的交互式图表
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='sin(x)'))
fig.update_layout(
    title='Sine Curve',
    xaxis_title='x',
    yaxis_title='y',
    legend=dict(x=0, y=1, traceorder='normal')
)
fig.show()