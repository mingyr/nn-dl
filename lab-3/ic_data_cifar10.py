import os
import numpy as np
import tensorflow as tf
import sonnet as snt 


class Input(snt.AbstractModule):
    def __init__(self, batch_size, image_dims, num_epochs = -1, name = 'input'):
        '''
        Args:
            batch_size: number of tfrecords to dequeue
            value_shape: the expected shape of values
        '''
        super(Input, self).__init__(name = name)
        self._batch_size = batch_size
        self._image_dims = image_dims
        self._num_epochs = num_epochs

    def _parse_function(self, example):
        dims = np.prod(self._image_dims)

        features = {
            "image": tf.FixedLenFeature([dims], dtype = tf.float32),
            "label": tf.FixedLenFeature([], dtype = tf.int64)
        }

        example_parsed = tf.parse_single_example(serialized = example, features = features)
        value = tf.reshape(example_parsed['image'], self._image_dims)

        label = example_parsed['label']

        return value, label
        
    def _build(self, filename):
        assert os.path.isfile(filename), "invalid file name: {}".format(filename)

        dataset = tf.data.TFRecordDataset([filename])
        dataset = dataset.map(self._parse_function)

        dataset = dataset.batch(self._batch_size)
        dataset = dataset.repeat(self._num_epochs)

        it = dataset.make_one_shot_iterator()
        images, labels = it.get_next()

        return images, labels


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

    
if __name__ == '__main__':
    num_images = 64
    image_width = 32
    image_height = 32   
    image_channels = 3
    
    input_ = Input(num_images, [image_height, image_width, image_channels])
    images, labels = input_('data/cifar-10/cifar-10-train.tfr')
    
    if not os.path.exists("output-data"):
        os.makedirs("output-data")
        
    writer = tf.summary.FileWriter("output-data", tf.get_default_graph())
        
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            
        image_str = draw_image(images, 8, 8)
        writer.add_summary(image_str, global_step = 0)


