import os
import numpy as np
import tensorflow as tf

from ic_data import Input
from ic_model import Model 

batch_size = 32

image_width = 28
image_height = 28
num_classes = 10

# 执行自动微分算法（或反向传播算法）的优化器参数
learning_rate = 0.001
lr_decay_steps = 100
lr_decay_factor = 0.9

# 训练时迭代次数
iterations = 1000

if not os.path.exists("output-train"):
    os.makedirs("output-train")

# 训练好的模型的保存路径
checkpoint_path = os.path.join("output-train", "model.ckpt")

# 以数据集方式加载样本，方便进行mini-batch训练
input_ = Input(batch_size, [image_height, image_width, 1])
images, labels = input_("data/mnist/mnist-train.tfr")

# 由输入得到模型的输出
net = Model(num_classes)
logits = net(images)

# 计算损失函数
labels = tf.one_hot(labels, num_classes, axis = -1)

with tf.control_dependencies([tf.assert_equal(tf.rank(labels), tf.rank(logits))]):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = logits)

loss = tf.reduce_mean(loss, name = 'loss')
loss_summary = tf.summary.scalar('loss', loss)      

# 设置学习率
global_step = tf.train.get_or_create_global_step()
lr = tf.train.exponential_decay(learning_rate, global_step, 
    lr_decay_steps, lr_decay_factor, staircase = True)

lr_summary = tf.summary.scalar('lr', lr)      

# 创建优化器
opt = tf.train.AdamOptimizer(lr)

# 进行优化
train_op = opt.minimize(loss, global_step = global_step, var_list = tf.trainable_variables())
          
# 合并所有summary信息          
all_summaries = tf.summary.merge_all() 
       
writer = tf.summary.FileWriter('output-train', tf.get_default_graph())

# 保存训练好的模型
saver = tf.train.Saver(tf.trainable_variables())

# 创建Session，进行训练
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    
    for i in range(iterations):
        loss_val, _, summ_str = sess.run([loss, train_op, all_summaries])
        print("{}-th iteration with loss {}".format(i, loss_val))
        writer.add_summary(summ_str, global_step = i)
    
    
    # 训练完成，保存模型
    print('Saving model.')
    saver.save(sess, checkpoint_path)
    print('Training complete')

    writer.close()
    