from utils import *
from imresize import imresize
from gkernel import generate_kernel

class dataGenerator(object):
    def __init__(self, output_shape, meta_batch_size, task_batch_size, tfrecord_path):
        self.buffer_size=1000 # tf.data.TFRecordDataset buffer size

        self.TASK_BATCH_SIZE=task_batch_size
        self.HEIGHT, self.WIDTH, self.CHANNEL=output_shape

        self.META_BATCH_SIZE=meta_batch_size
        self.tfrecord_path = tfrecord_path
        self.label_train = self.load_tfrecord()

    def make_data_tensor(self, sess, scale_list, noise_std=0.0):
        label_train_=sess.run(self.label_train)

        input_meta =[]
        label_meta =[]

        for t in range(self.META_BATCH_SIZE):
            input_task = []
            label_task = []

            scale = np.random.choice(scale_list, 1)[0]
            Kernel = generate_kernel(k1=scale*2.5, ksize=15)
            for idx in range(self.TASK_BATCH_SIZE*2):
                img_HR=label_train_[t*self.TASK_BATCH_SIZE*2 + idx]
                clean_img_LR=imresize(img_HR,scale=1./scale, kernel=Kernel)

                img_LR=np.clip(clean_img_LR+ np.random.randn(*clean_img_LR.shape)*noise_std, 0., 1.)

                img_ILR=imresize(img_LR, scale=scale, output_shape=img_HR.shape, kernel='cubic')

                input_task.append(img_ILR)
                label_task.append(img_HR)

            input_meta.append(np.asarray(input_task))
            label_meta.append(np.asarray(label_task))

        input_meta=np.asarray(input_meta)
        label_meta=np.asarray(label_meta)

        inputa=input_meta[:,:self.TASK_BATCH_SIZE,:,:]
        labela=label_meta[:,:self.TASK_BATCH_SIZE,:,:]
        inputb=input_meta[:,self.TASK_BATCH_SIZE:,:,:]
        labelb=label_meta[:,self.TASK_BATCH_SIZE:,:,:]

        return inputa, labela, inputb, labelb

    '''Load TFRECORD'''
    def _parse_function(self, example_proto):
        keys_to_features = {'label': tf.FixedLenFeature([], tf.string)}

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)

        img = parsed_features['label']
        img = tf.divide(tf.cast(tf.decode_raw(img, tf.uint8), tf.float32), 255.)
        img = tf.reshape(img, [self.HEIGHT, self.WIDTH, self.CHANNEL])

        return img


    def load_tfrecord(self):
        dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        dataset = dataset.map(self._parse_function)

        dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.TASK_BATCH_SIZE*self.META_BATCH_SIZE*2)
        iterator = dataset.make_one_shot_iterator()

        label_train = iterator.get_next()

        return label_train