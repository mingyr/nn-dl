import os
import numpy as np
import tensorflow as tf
from mnist_data import Dataset
from tensorboard.plugins import projector

image_width = 28
image_height = 28
image_channels = 1
num_classes = 10

log_dir = "output-test"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
# 模型加载函数
def load_model(save_path):
    assert os.path.exists(save_path), "模型路径不存在"

    loaded = tf.saved_model.load(save_path)
    assert len(loaded.all_variables) > 0, "加载模型失败"

    return loaded

# 加载测试数据与模型
ds = Dataset(1, [image_height, image_width, image_channels], 1)
model = load_model("saved_model")

embeddings = []
metadata = os.path.join(log_dir, 'metadata.tsv')
metadata_file = open(metadata, 'w')

accuracy = 0
count = 1000
for i, (image, label) in enumerate(ds(r"data\mnist\mnist-test.tfr")):
    logit = model.inference(image)
    logit = tf.squeeze(logit, axis=0)
    embeddings.append(logit)
    num = tf.argmax(logit, axis=-1)
    metadata_file.write('%d\n' % label[0])
    if num == label:
        accuracy = accuracy + 1
    if (count == 1):
        break
    else:
        count = count - 1
        
metadata_file.close()

accuracy = accuracy / (i + 1)
print("The accuracy of inference on test set is {}".format(accuracy))

embedding_var = tf.Variable(embeddings, dtype=tf.float32)
checkpoint = tf.train.Checkpoint(embedding=embedding_var)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE" # embedding_var.name
embedding.sprite.image_path = 'mnist_10k_sprite.png' # Path relative to the log directory
embedding.sprite.single_image_dim.extend([image_height, image_width])
embedding.metadata_path = 'metadata.tsv'             # Path relative to the log directory

projector.visualize_embeddings(log_dir, config)    
