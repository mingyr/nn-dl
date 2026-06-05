import os
import numpy as np
import tensorflow as tf
import sonnet as snt
import tfmpl

class Dataset(snt.Module):
    def __init__(self, batch_size, image_dims, num_epochs=-1, name='dataset'):
        super(Dataset, self).__init__(name=name)
        self._batch_size = batch_size
        self._image_dims = image_dims
        self._num_epochs = num_epochs

    def _parse_function(self, example):
        dims = np.prod(self._image_dims)

        features = {
            "image": tf.io.FixedLenFeature([dims], dtype=tf.float32),
            "label": tf.io.FixedLenFeature([], dtype=tf.int64) 
        }

        example_parsed = tf.io.parse_single_example(serialized=example, features=features)
        value = tf.reshape(example_parsed['image'], self._image_dims)
        label = example_parsed['label']

        return value, label
        
    def __call__(self, filename):
        assert os.path.isfile(filename), "invalid file name: {}".format(filename)

        dataset = tf.data.TFRecordDataset([filename])
        dataset = dataset.map(self._parse_function)
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.repeat(self._num_epochs)

        return dataset


@tfmpl.figure_tensor
def draw_image(images, rows, cols):
    num_figs = len(images)
    fig = tfmpl.create_figure(figsize= (12.8, 12.8))

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

            ax.imshow(images[seq-1, ...])
            ax.axis('off')

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    num_images = 64
    image_width = 28
    image_height = 28   
    image_channels = 1
 
    ds = Dataset(num_images, [image_height, image_width, image_channels])
    images, labels = next(iter(ds(r'data\mnist\mnist-train.tfr')))
    
    log_dir = "output-data"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    summary_writer = tf.summary.create_file_writer(log_dir)
    with summary_writer.as_default():
        image_tensor = draw_image(images, 8, 8)
        image_summary = tf.summary.image("images", image_tensor, step=0)    
    summary_writer.close()    
