import os
import numpy as np
import tensorflow as tf
import cifar10

num_images = 64
image_width = 32
image_height = 32

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


if not os.path.exists("output-data"):
    os.makedirs("output-data")
    
writer = tf.summary.FileWriter("output-data", tf.get_default_graph())    

if not os.path.exists("data/CIFAR-10/"):
    os.makedirs("data/CIFAR-10/")

cifar10.extract()
images, cls = cifar10.load_test_data()

images = tf.constant(images)
    
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    images_val = sess.run(tf.gather(images, list(range(num_images))))
        
    image_str = draw_image(images_val, 8, 8)
    writer.add_summary(image_str, global_step = 0)




