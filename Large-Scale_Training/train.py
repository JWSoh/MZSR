import time
import model
from imresize import imresize
from utils import *

class Train(object):
    def __init__(self, trial, step, size, batch_size, learning_rate, max_epoch, tfrecord_path, checkpoint_dir, scale,num_of_data, conf, model_num):

        print('Initialize Training')
        self.trial=trial
        self.step=step
        self.HEIGHT = size[0]
        self.WIDTH = size[1]
        self.CHANNEL = size[2]
        self.BATCH_SIZE = batch_size
        self.learning_rate=learning_rate
        self.EPOCH = max_epoch
        self.tfrecord_path = tfrecord_path
        self.checkpoint_dir=checkpoint_dir
        self.scale= scale
        self.num_of_data=num_of_data
        self.conf=conf
        self.model_num=model_num

        self.input = tf.placeholder(dtype=tf.float32,shape=[None,None,None,self.CHANNEL])
        self.label = tf.placeholder(dtype=tf.float32,shape=[None,None,None,self.CHANNEL])

        self.MODEL = model.MODEL('MODEL')
        self.PARAM = model.Weights('MODEL')
        self.MODEL.forward(self.input, self.PARAM.weights)

    def calc_loss(self):
        self.loss=tf.losses.absolute_difference(self.MODEL.output , self.label)

    def run(self):
        print('Setting Train Configuration')

        self.calc_loss()

        '''Learning rate and the optimizer'''
        self.global_step=tf.Variable(self.step, name='global_step', trainable=False)

        self.lr=tf.train.exponential_decay(self.learning_rate,self.global_step, 100000, 0.5, staircase=True)
        self.lr=tf.maximum(self. lr, 1e-4)

        self.opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=self.global_step)


        '''Summary operator for Tensorboard'''
        self.summary_op=tf.summary.merge([tf.summary.scalar('loss', self.loss),
                                          tf.summary.image('1. Input', tf.clip_by_value(self.MODEL.input, 0., 1.),max_outputs=4),
                                          tf.summary.image('2. Output',tf.clip_by_value(self.MODEL.output,0., 1.),max_outputs=4),
                                          tf.summary.image('3. Label', tf.clip_by_value(self.label, 0., 1.), max_outputs=4)])

        '''Training'''
        self.saver = tf.train.Saver(max_to_keep=10000)
        self.init = tf.global_variables_initializer()

        label_train, input_train = self.load_tfrecord()

        count_param()

        t2 = time.time()

        with tf.Session(config=self.conf) as sess:
            sess.run(self.init)

            could_load = load(self.saver, sess, self.checkpoint_dir, folder='Model%d' % self.trial)
            if could_load:
                print('Iteration:', self.step)
                print(' =========== Load Succeeded ============')
            else:
                print(" ========== No model to load ===========")

            train_writer = tf.summary.FileWriter('./logs%d' % self.trial, sess.graph)

            print('Training Starts')

            step = self.step

            num_of_batch = self.num_of_data // self.BATCH_SIZE
            s_epoch = (step * self.BATCH_SIZE) // self.num_of_data


            epoch=s_epoch
            while True:
                try:
                    label_train_, input_train_ = sess.run([label_train, input_train])

                    input_bic_=np.zeros(label_train_.shape)

                    for idx in range(len(label_train_)):
                        input_bic_[idx]=imresize(input_train_[idx], scale=self.scale, kernel='cubic')


                    sess.run(self.opt, feed_dict={self.input: input_bic_, self.label: label_train_})

                    step = step + 1

                    if step % 1000 == 0:
                        t1 = t2
                        t2 = time.time()
                        loss_, summary = sess.run([self.loss, self.summary_op],
                            feed_dict={self.input: input_bic_, self.label: label_train_})

                        print('Iteration:', step, 'Loss:', loss_, 'LR:', sess.run(self.lr), 'Time: %.2f' % (t2-t1))

                        train_writer.add_summary(summary, step)
                        train_writer.flush()

                    if step % 10000 == 0:
                        save(self.saver, sess, self.checkpoint_dir, self.trial, step)

                    if step == 800001:
                        print('Training Done')
                        break


                except KeyboardInterrupt:
                    print('***********KEY BOARD INTERRUPT *************')
                    print('Epoch:', epoch, 'Iteration:', step)
                    save(self.saver, sess, self.checkpoint_dir, self.trial, step)
                    break


    '''Load TFRECORD'''
    def _parse_function(self, example_proto):
        keys_to_features = {'label': tf.FixedLenFeature([], tf.string),
                            'image': tf.FixedLenFeature([], tf.string)}

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)

        img = parsed_features['image']
        img = tf.divide(tf.cast(tf.decode_raw(img, tf.uint8), tf.float32), 255.)
        img=  tf.reshape(img,[self.HEIGHT//self.scale,self.WIDTH//self.scale,self.CHANNEL])

        label = parsed_features['label']
        label = tf.divide(tf.cast(tf.decode_raw(label, tf.uint8), tf.float32), 255.)
        label = tf.reshape(label, [self.HEIGHT, self.WIDTH, self.CHANNEL])

        return label, img

    def load_tfrecord(self):
        dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        dataset = dataset.map(self._parse_function)

        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.BATCH_SIZE)
        iterator = dataset.make_one_shot_iterator()

        label_train, input_train = iterator.get_next()

        return label_train, input_train
