import os
import numpy as np
import tensorflow as tf
import tfmpl

from ic_data import Input
from ic_model import Model 

from tensorflow.contrib.tensorboard.plugins import projector

image_width = 32
image_height = 32
num_classes = 10

LOG_DIR = "output-test"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    
if not os.path.exists("output-train"):
    raise ValueError("Non-existing output-train folder")

# 训练好的模型的保存路径
checkpoint_dir = "output-train"

graph_test = tf.Graph()
with graph_test.as_default():
    # 加载测试数据
    input_ = Input(1, [image_height, image_width, 3], 1)
    image, label = input_("data/cifar-10/cifar-10-test.tfr")

    # 由输入得到模型的输出
    net = Model(num_classes)
    logit = net(image)
    
    logit = tf.squeeze(logit, axis = 0)
    id = tf.argmax(logit, axis = -1)
    
    restorer = tf.train.Saver()

embeddings = []
metadata = os.path.join(LOG_DIR, 'metadata.tsv')
metadata_file = open(metadata, 'w')
count = 0

desc = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

with tf.Session(graph = graph_test) as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint.
        restorer.restore(sess, ckpt.model_checkpoint_path)

        print('Successfully loaded model from %s.' % ckpt.model_checkpoint_path)

    else:
        print('No checkpoint file found')
        exit

    while True:
        try:
            logit_val, id_val = sess.run([logit, id])
            embeddings.append(logit_val)
            metadata_file.write('{}\n'.format(desc[id_val]))
            # sess.run(metric_update)
            count = count + 1    
        except tf.errors.OutOfRangeError:
            break
    
    metadata_file.flush()
    metadata_file.close()


graph_visualize = tf.Graph()
with graph_visualize.as_default():
    embedding_var = tf.Variable(np.zeros([count, num_classes]), dtype = tf.float32, name="embedding_var")
    embedding_op = embedding_var.assign(tf.convert_to_tensor(np.array(embeddings), dtype = tf.float32))

    saver = tf.train.Saver([embedding_var])
    
    writer = tf.summary.FileWriter(LOG_DIR, graph_visualize)
    config = projector.ProjectorConfig()

    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.sprite.image_path = 'cifar10_sprite.png' # Path relative to writer's log directory
    embedding.sprite.single_image_dim.extend([32, 32])
    embedding.metadata_path = 'metadata.tsv'             # Path relative to writer's log directory

    projector.visualize_embeddings(writer, config)    

with tf.Session(graph = graph_visualize) as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    sess.run(embedding_op)
    
    saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'), global_step = 1)
    