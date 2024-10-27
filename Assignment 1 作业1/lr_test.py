import os
import numpy as np
import tensorflow as tf
import tfmpl
from lr_model import LRModel 

log_dir = "output-test"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
# 训练好的模型的保存路径   
save_path = "saved_model" 
if not os.path.exists(save_path):
    raise ValueError("Cannot find saved model")

# 生成测试数据，为使对比更加明显，
# 我们假设数据不含噪声。
x = tf.random.uniform((100, 1)) * 4 + 2.0
y_d = 1.6 * x + 0.4

# 模型加载函数
def load_model(save_path):
    assert os.path.exists(save_path), "模型路径不存在"

    loaded = tf.saved_model.load(save_path)
    assert len(loaded.all_variables) > 0, "加载模型失败"

    return loaded

@tfmpl.figure_tensor
def draw_scatter(data, titles, colors): 
    '''Draw scatter plots. One for each color.'''  
    fig = tfmpl.create_figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(data[:, 0], data[:, 1], c=colors[0])
    ax1.set_title(titles[0])
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(data[:, 0], data[:, 2], c=colors[1])
    ax2.set_title(titles[1])
    fig.tight_layout()
    return fig

def main(sample_index = None):
    model = load_model(save_path)
    y = model.inference(x)
    pts = tf.concat([x, y_d, y], axis=-1)
    
    summary_writer = tf.summary.create_file_writer(log_dir)
    with summary_writer.as_default():
        image_tensor = draw_scatter(pts, ["Ground-truth Model", "Learned Model"], ['b', 'r'])
        image_summary = tf.summary.image("images", image_tensor, step=0)    
    summary_writer.close()
    
if __name__ == "__main__":
    main()    
