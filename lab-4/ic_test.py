import os
import numpy as np
import tensorflow as tf
import tfmpl

from ic_data import Input
from ic_model import Encoder, Decoder 

from tensorflow.contrib.tensorboard.plugins import projector

image_width = 28
image_height = 28
num_classes = 10

LOG_DIR = "output-test"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    
if not os.path.exists("output-train"):
    raise ValueError("Non-existing output-train folder")

# 训练好的模型的保存路径
checkpoint_dir = "output-train"

input_ = Input(16, [image_height, image_width, 1], 1)
images, _ = input_("data/MNIST/mnist-test.tfr")
    
# 由输入得到模型的输出
encoder = Encoder()
decoder = Decoder()

latents = encoder(images)
restored = decoder(latents)

images = tf.concat(images, axis=0)
restored = tf.concat(restored, axis=0)

saver = tf.train.Saver()
writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())   
 
def draw_image(images, rows, cols, tensor_name = "images"):
    import tfmpl

    @tfmpl.figure_tensor
    def draw(images):
        num_figs = len(images)
        fig = tfmpl.create_figures(1, figsize= (12.8, 12.8))[0]

        # pdb.set_trace()
        for i in range(rows):
            for j in range(cols):
                seq = i * cols + j + 1
                if seq > num_figs:
                    fig.tight_layout()
                    return fig

                if num_figs == 1:
                    ax = fig.add_subplot(1, 1, 1)
                else:
                    ax = fig.add_subplot(rows, cols, seq)

                ax.axis('off')
                ax.imshow(images[seq-1, ...])

        fig.tight_layout()
        return fig

    image_tensor = draw(images)
    image_summary = tf.summary.image(tensor_name, image_tensor)
    sess = tf.get_default_session()
    assert sess != None, "Invalid session"
    image_str = sess.run(image_summary)

    return image_str


with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
        print('No checkpoint file found')
        exit

    images_val, restored_val = sess.run([images, restored])
    
    images_str = draw_image(images_val, 4, 4)
    writer.add_summary(images_str, global_step = 0)

    images_str = draw_image(restored_val, 4, 4)
    writer.add_summary(images_str, global_step = 1)
    
    writer.close()