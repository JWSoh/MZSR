import os
import tensorflow as tf
import imageio
import numpy as np
import math

def imread(path):
    img = imageio.imread(path).astype(np.float32)
    img=img/255.
    return img

def load(saver, sess, checkpoint_dir, folder):
    print(" ========== Reading Checkpoints ============")
    checkpoint=os.path.join(checkpoint_dir, folder)

    ckpt = tf.train.get_checkpoint_state(checkpoint)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint, ckpt_name))

        print(" ============== Success to read {} ===============".format(ckpt_name))
        return True
    else:
        print(" ============= Failed to find a checkpoint =============")
        return False

def save(saver, sess, checkpoint_dir, trial, step):
    model_name="model"
    checkpoint=os.path.join(checkpoint_dir, "Model%d" % trial)

    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    saver.save(sess,os.path.join(checkpoint,model_name),global_step=step)

def psnr(img1, img2):
    img1=np.float64(img1)
    img2=np.float64(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    if np.max(img1) <= 1.0:
        PIXEL_MAX= 1.0
    else:
        PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def modcrop(imgs, modulo):
    sz=imgs.shape
    sz=np.asarray(sz)
    if len(sz)==2:
        sz = sz - sz% modulo
        out = imgs[0:sz[0], 0:sz[1]]
    elif len(sz)==3:
        szt = sz[0:2]
        szt = szt - szt % modulo
        out = imgs[0:szt[0], 0:szt[1],:]

    return out

def count_param(scope=None):
    N=np.sum([np.prod(v.get_shape().as_list()) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)])
    print('Model Params: %d K' % (N/1000))