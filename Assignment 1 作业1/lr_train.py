import os
import numpy as np
import tensorflow as tf
import sonnet as snt
from lr_model import LRModel 

# 执行自动微分算法（或反向传播算法）的优化器参数
learning_rate = 0.01

# 记录训练信息
log_dir = "output-train"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
summary_writer = tf.summary.create_file_writer(log_dir)

# 因为问题比较简单，直接生成训练与验证样本
# 包含输入数据与标签，标签加噪声模拟实际情况    
x_train = np.random.uniform(size=(100, 1)) * 4
y_train = 1.6 * x_train + 0.4 + tf.random.normal(tf.shape(x_train), stddev=0.1)

# 将输入包装成数据集与输入管线，方便进行mini-batch训练
ds_train = tf.data.Dataset.from_tensor_slices(tf.concat([x_train, y_train], axis=-1))
ds_train = ds_train.shuffle(buffer_size=100).batch(16).repeat(1)
ds_train = ds_train.map(lambda s: tf.split(s, num_or_size_splits=2, axis=-1))

x_val = np.random.uniform(size=(40, 1)) * 4
y_val = 1.6 * x_val + 0.4 + tf.random.normal(tf.shape(x_val), stddev=0.1)

# 将输入包装成数据集与输入管线，方便进行验证
ds_val = tf.data.Dataset.from_tensor_slices(tf.concat([x_val, y_val], axis=-1))
ds_val = ds_val.batch(8).repeat(-1)
ds_val = ds_val.map(lambda s: tf.split(s, num_or_size_splits=2, axis=-1))

# 建立模型与优化器
model = LRModel()
opt = snt.optimizers.Adam(learning_rate)

# 定义单步训练
def train(x, labels, step):
    # 记录前向传播信息
    with tf.GradientTape() as tape:
        y = model(x)
        loss = tf.math.reduce_mean(
            tf.math.squared_difference(y, labels))
  
    # 计算梯度，并更新权重
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    opt.apply(gradients, variables)

    with summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=step)

    return loss
    
# 验证模型的超参
def validate(dataset_iter, step):
    x, labels = next(dataset_iter)
    y = model(x)
    err = tf.math.reduce_mean(
        tf.math.squared_difference(y, labels))
  
    with summary_writer.as_default():
        tf.summary.scalar('error', err, step=step)

    return err

# 保存模型
def save_model(module):
    feature_size = 1
    @tf.function(input_signature=[tf.TensorSpec([None, feature_size])])
    def inference(x):
        return module(x)

    save_path = r"saved_model"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    to_save = snt.Module()
    to_save.inference = inference
    to_save.all_variables = list(module.variables)
    tf.saved_model.save(to_save, save_path)

# 训练模型
# @tf.function
def main():
    num_epochs = 20
    step = np.int64(0)
    it_val = iter(ds_val)
    for _ in range(num_epochs):
        for x, labels in ds_train:
            step = step + 1
            loss = train(x, labels, step)
            err = validate(it_val, step)
            tf.print("iteration:", step, " loss - ", loss, " error - ", err)
    
if __name__ == "__main__":
    main()        

    # 保存模型        
    save_model(model)
