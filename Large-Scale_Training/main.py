import train
from utils import *
from argparse import ArgumentParser

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

conf = tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction = 0.9

HEIGHT = 96
WIDTH = 96
CHANNEL = 3
BATCH_SIZE = 32
EPOCH = 20000
LEARNING_RATE = 4e-4
CHECK_POINT_DIR = 'SR'
SCALE = 2

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--trial', type=int,
            dest='trial', help='Trial Number',
            metavar='trial', default=0)
    parser.add_argument('--gpu',
            dest='gpu_num', help='GPU Number',
            metavar='GPU_NUM', default='0')
    parser.add_argument('--step', type=int,
            dest='global_step', help='Global Step',
            metavar='GLOBAL_STEP', default=0)

    return parser

def main():
    parser = build_parser()
    options = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu_num

    NUM_OF_DATA = 640000
    TF_RECORD_PATH=['../train_SR_bicubic_X2.tfrecord']

    Trainer=train.Train(trial=options.trial,step=options.global_step,size=[HEIGHT,WIDTH,CHANNEL], batch_size=BATCH_SIZE,
                        learning_rate=LEARNING_RATE, max_epoch=EPOCH,tfrecord_path=TF_RECORD_PATH,checkpoint_dir=CHECK_POINT_DIR,
                        scale=SCALE,num_of_data=NUM_OF_DATA, conf=conf)
    Trainer.run()

if __name__ == '__main__':
    main()