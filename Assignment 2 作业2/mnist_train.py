import os
import numpy as np
import tensorflow as tf
import sonnet as snt
from mnist_data import Dataset
from mnist_model import Model 

batch_size = 32
image_width = 28
image_height = 28
image_channels = 1
num_classes = 10

# 建立模型与优化器
model = Model(num_classes)
learning_rate = 0.01
optimizer = snt.optimizers.Adam(learning_rate)

# 准备存储运行时信息的文件夹与对象
log_dir = "output-train"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

summary_writer = tf.summary.create_file_writer(log_dir)

# 定义单步训练
def train(images, labels, step):
    # 记录前向传播信息
    with tf.GradientTape() as tape:
        logits = model(images)
        labels = tf.one_hot(labels, depth=num_classes)
        with tf.control_dependencies([tf.assert_equal(tf.shape(logits), tf.shape(labels))]):
            loss = tf.math.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels, logits))
  
    # 计算梯度，并更新权重
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply(gradients, variables)

    with summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=step)

    return loss

    
# 验证模型的超参
def validate(dataset_iter, step):
    images, labels = next(dataset_iter)
    logits = model(images)
    logits = tf.math.argmax(logits, -1)
    with tf.control_dependencies([tf.assert_equal(tf.shape(logits), tf.shape(labels))]):
        prediction = tf.equal(labels, logits)
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
  
    with summary_writer.as_default():
        tf.summary.scalar('accuracy', accuracy, step=step)

    return accuracy
    
# 保存模型
def save_model(module):
    @tf.function(input_signature=[tf.TensorSpec([None, image_height, image_width, image_channels])])
    def inference(x):
        return module(x)

    save_path = r"saved_model"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    to_save = snt.Module()
    to_save.inference = inference
    to_save.all_variables = list(module.variables)
    tf.saved_model.save(to_save, save_path)


def main():
    # 加载准备好的数据
    train_data_path = r"data\mnist\mnist-train.tfr"
    val_data_path = r"data\mnist\mnist-test.tfr"

    train_dataset = Dataset(batch_size, 
        [image_height, image_width, image_channels], 1)(train_data_path)
    val_dataset = Dataset(batch_size, 
        [image_height, image_width, image_channels])(val_data_path)

    @tf.function
    def loop():
        num_epochs = 2
        step = np.int64(0)
        val_dataset_iter = iter(val_dataset)
        for _ in range(num_epochs):
            for features, labels in train_dataset:
                step = step + 1
                loss = train(features, labels, step, )
                accu = validate(val_dataset_iter, step)
                tf.print("iteration:", step, " loss - ", loss, " accuracy - ", accu)
                
    # 训练模型
    loop()
    
    # 保存模型        
    save_model(model)
    
    
if __name__ == "__main__":
    main()        

