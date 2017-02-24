import tensorflow as tf

import os
import dateutil.tz
import datetime
from  utils import mkdir_p
from utils import MnistData
from InfoGan import InfoGan

flags = tf.app.flags

flags.DEFINE_integer("OPER_FLAG" , 3 , "the flag of  opertion")
flags.DEFINE_integer("extend" , 1 , "contional value y")

FLAGS = flags.FLAGS

if __name__ == "__main__":

    root_log_dir = "/tmp/logs/mnist_test"
    root_checkpoint_dir = "/tmp/gan_model/gan_model.ckpt"
    encode_z_checkpoint_dir = "/tmp/encode_z_model/encode_model.ckpt"
    encode_y_checkpoint_dir = "/tmp/encode_y_model/encode_model.ckpt"
    sample_path = "sample/mnist_gan"

    OPER_FLAG = FLAGS.OPER_FLAG

    if OPER_FLAG == 0:

        build_model_flag = 0

    elif OPER_FLAG == 1:

        build_model_flag = 1

    elif OPER_FLAG == 2:

        build_model_flag = 2

    elif OPER_FLAG == 3:

        build_model_flag = 3

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    batch_size = 64
    max_epoch = 20

    #for mnist train 62 + 2 + 10
    sample_size = 64
    dis_learn_rate = 0.0002
    gen_learn_rate = 0.0002

    exp_name = "mnist_%s" % timestamp

    log_dir = os.path.join(root_log_dir, exp_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

    mkdir_p(log_dir)
    mkdir_p(checkpoint_dir)
    mkdir_p(sample_path)
    mkdir_p(encode_z_checkpoint_dir)
    mkdir_p(encode_y_checkpoint_dir)

    data , label = MnistData().load_mnist()

    infoGan = InfoGan(batch_size=batch_size, max_epoch=max_epoch, build_model_flag = build_model_flag,
                      model_path=root_checkpoint_dir, encode_z_model=encode_z_checkpoint_dir,encode_y_model=encode_y_checkpoint_dir,
                      data=data,label=label, extend_value=FLAGS.extend,
                      network_type="mnist", sample_size=sample_size,
                      sample_path=sample_path , log_dir = log_dir , gen_learning_rate=gen_learn_rate, dis_learning_rate=dis_learn_rate , info_reg_coeff=1.0)

    #start the training

    if OPER_FLAG == 0:

        print "Training Gan"
        infoGan.train()

    elif OPER_FLAG == 1:

        print "Training encode for z"
        infoGan.train_ez()

    elif OPER_FLAG == 2:

        print "Training encode for Y"
        infoGan.train_ey()

    else:

        print "This is Test"
        infoGan.test()







