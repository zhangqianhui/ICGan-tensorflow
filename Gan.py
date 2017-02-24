import tensorflow as tf
from ops import batch_normal, lrelu, de_conv, conv2d, fully_connect, conv_cond_concat
from utils import MnistData
from utils import save_images
from utils import sample_label
import numpy as np
import cv2


TINY = 1e-8

class Gan(object):

    #build model
    def __init__(self, batch_size, max_epoch, build_model_flag, model_path, encode_z_model, encode_y_model, data, label, extend_value,
                 network_type , sample_size, sample_path , log_dir , gen_learning_rate , dis_learning_rate , info_reg_coeff):

        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.model_path = model_path
        self.encode_z_model = encode_z_model
        self.encode_y_model = encode_y_model
        self.ds_train = data
        self.label_y = label
        self.extend_value = extend_value
        self.type = network_type
        self.sample_size = sample_size
        self.sample_path = sample_path
        self.log_dir = log_dir
        self.learning_rate_gen = gen_learning_rate
        self.learning_rate_dis = dis_learning_rate
        self.info_reg_coeff = info_reg_coeff
        self.log_vars = []
        #self.output_dist= MeanBernoulli(28*28)
        self.channel = 1
        self.y_dim = 10

        self.output_size = MnistData().image_size
        self.build_model = build_model_flag

        self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.sample_size])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim])

        self.weights1, self.biases1 = self.get_gen_variables()
        self.weights2, self.biases2 = self.get_dis_variables()

        if self.build_model == 0:
            self.build_model1()
        elif self.build_model == 1:
            self.build_model2()
        elif self.build_model == 2:
            self.build_model3()
        else:
            self.build_model4()

    def build_model1(self):

        #Constructing the Gan
        #Get the variables

        self.fake_images = self.generate(self.z, self.y, weights=self.weights1, biases=self.biases1)

        # the loss of dis network
        self.D_pro = self.discriminate(self.images, self.y, self.weights2, self.biases2, False)

        self.G_pro = self.discriminate(self.fake_images, self.y, self.weights2, self.biases2, True)

        self.G_fake_loss = -tf.reduce_mean(tf.log(self.G_pro + TINY))
        self.loss = -tf.reduce_mean(tf.log(1. - self.G_pro + TINY) + tf.log(self.D_pro + TINY))

        self.log_vars.append(("generator_loss", self.G_fake_loss))
        self.log_vars.append(("discriminator_loss", self.loss))

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]

        self.saver = tf.train.Saver(self.g_vars)

        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    #Training the Encode_z
    def build_model2(self):

        self.weights3, self.biases3 = self.get_en_z_variables()

        #training Ez

        self.fake_images = self.generate(self.z, self.y, weights=self.weights1, biases=self.biases1)
        self.e_z= self.encode_z(self.fake_images, weights=self.weights3, biases=self.biases3)

        self.loss_z = tf.reduce_mean(tf.square(tf.contrib.layers.flatten(self.e_z - self.z)))

        t_vars = tf.trainable_variables()

        self.g_vars = [var for var in t_vars if 'gen' in var.name]
        self.enz_vars = [var for var in t_vars if 'enz' in var.name]

        print len(self.g_vars)
        print len(self.enz_vars)

        self.saver = tf.train.Saver(self.g_vars)
        self.saver_z = tf.train.Saver(self.g_vars + self.enz_vars)

    #Training the Encode_y
    def build_model3(self):

        self.weights4, self.biases4 = self.get_en_y_variables()
        # Training Ey
        self.e_y = self.encode_y(self.images, weights=self.weights4, biases=self.biases4)

        self.loss_y = tf.reduce_mean(tf.square(self.e_y - self.y))

        t_vars = tf.trainable_variables()

        self.eny_vars = [var for var in t_vars if 'eny' in var.name]

        self.saver_y = tf.train.Saver(self.eny_vars)

    #Test model
    def build_model4(self):

        self.weights3, self.biases3 = self.get_en_z_variables()
        self.weights4, self.biases4 = self.get_en_y_variables()

        self.e_z = self.encode_z(self.images, weights=self.weights3, biases=self.biases3)
        self.e_y = self.encode_y(self.images, weights=self.weights4, biases=self.biases4)

        #Changing y : + 1 or +2 or +3
        self.e_y = tf.one_hot(tf.arg_max(self.e_y, 1) + self.extend_value, 10)

        self.fake_images = self.generate(self.e_z, self.e_y, weights=self.weights1, biases=self.biases1)

        t_vars = tf.trainable_variables()

        self.g_vars = [var for var in t_vars if 'gen' in var.name]
        self.enz_vars = [var for var in t_vars if 'enz' in var.name]
        self.eny_vars = [var for var in t_vars if 'eny' in var.name]

        self.saver = tf.train.Saver(self.g_vars)

        self.saver_z = tf.train.Saver(self.g_vars + self.enz_vars)
        self.saver_y = tf.train.Saver(self.eny_vars)

    #do train
    def train(self):

        opti_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate_dis, beta1=0.5).minimize(self.loss , var_list=self.d_vars)
        opti_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate_gen, beta1=0.5).minimize(self.G_fake_loss, var_list=self.g_vars)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init)

            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            #self.saver.restore(sess , self.model_path)

            batch_num = 0
            e = 0
            step = 0

            while e <= self.max_epoch:

                rand = np.random.randint(0, 100)
                rand = 0

                while batch_num < len(self.ds_train)/self.batch_size:

                    step = step + 1
                    realbatch_array, real_y = MnistData.getNextBatch(self.ds_train, self.label_y, rand, batch_num,self.batch_size)

                    batch_z = np.random.normal(0, 1 , size=[self.batch_size, self.sample_size])

                    #optimization D
                    _,summary_str = sess.run([opti_D, summary_op], feed_dict={self.images:realbatch_array, self.z: batch_z, self.y:real_y})
                    summary_writer.add_summary(summary_str , step)
                    #optimizaiton G
                    _,summary_str = sess.run([opti_G, summary_op], feed_dict={self.images:realbatch_array, self.z: batch_z, self.y:real_y})
                    summary_writer.add_summary(summary_str , step)
                    batch_num += 1

                    if step%1 ==0:

                        D_loss = sess.run(self.loss, feed_dict={self.images:realbatch_array, self.z: batch_z, self.y:real_y})
                        fake_loss = sess.run(self.G_fake_loss, feed_dict={self.z : batch_z, self.y:real_y})
                        print("EPOCH %d step %d: D: loss = %.7f G: loss=%.7f " % (e, step , D_loss, fake_loss))

                    if np.mod(step , 50) == 1:

                        sample_images = sess.run(self.fake_images ,feed_dict={self.z:batch_z, self.y:sample_label()})
                        save_images(sample_images[0:64] , [8, 8], './{}/train_{:02d}_{:04d}.png'.format(self.sample_path, e, step))
                        #Save the model
                        self.saver.save(sess , self.model_path)

                e += 1
                batch_num = 0

            save_path = self.saver.save(sess , self.model_path)
            print "Model saved in file: %s" % save_path

    def train_ez(self):

        opti_EZ = tf.train.AdamOptimizer(learning_rate = 0.01, beta1 = 0.5).minimize(self.loss_z,
                                                                                      var_list=self.enz_vars)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init)
            #summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            self.saver.restore(sess , self.model_path)

            batch_num = 0
            e = 0
            step = 0

            while e <= self.max_epoch:

                rand = np.random.randint(0, 100)
                rand = 0

                while batch_num < len(self.ds_train) / self.batch_size:

                    step = step + 1

                    _,label_y = MnistData.getNextBatch(self.ds_train, self.label_y, rand, batch_num,
                                                                     self.batch_size)
                    batch_z = np.random.normal(0, 1, size=[self.batch_size, self.sample_size])

                    # optimization E
                    sess.run(opti_EZ, feed_dict={self.y: label_y,self.z: batch_z})
                    batch_num += 1

                    if step % 10 == 0:

                        ez_loss = sess.run(self.loss_z, feed_dict={self.y: label_y,self.z: batch_z})
                        #summary_writer.add_summary(ez_loss, step)
                        print("EPOCH %d step %d EZ loss %.7f" % (e, step, ez_loss))

                    if np.mod(step, 50) == 0:

                        # sample_images = sess.run(self.fake_images, feed_dict={self.e_y:})
                        # save_images(sample_images[0:64], [8, 8],
                        #             './{}/train_{:02d}_{:04d}.png'.format(self.sample_path, e, step))
                        self.saver_z.save(sess, self.encode_z_model)

                e += 1
                batch_num = 0

            save_path = self.saver_z.save(sess, self.encode_z_model)
            print "Model saved in file: %s" % save_path

    def train_ey(self):

        opti_EY = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(self.loss_y,
                                                                                 var_list=self.eny_vars)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init)
            # summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            batch_num = 0
            e = 0
            step = 0

            while e <= self.max_epoch:

                rand = np.random.randint(0, 100)
                rand = 0

                while batch_num < len(self.ds_train) / self.batch_size:

                    step = step + 1

                    realbatch_image, label_y = MnistData.getNextBatch(self.ds_train, self.label_y, rand, batch_num,
                                                        self.batch_size)
                    #batch_z = np.random.normal(0, 1, size=[self.batch_size, self.sample_size])

                    # optimization E
                    sess.run(opti_EY, feed_dict={self.y: label_y, self.images: realbatch_image})
                    batch_num += 1

                    if step % 10 == 0:

                        ey_loss = sess.run(self.loss_y, feed_dict={self.y: label_y, self.images:realbatch_image})
                        #summary_writer.add_summary(ez_loss, step)
                        print("EPOCH %d step %d EY loss %.7f" % (e, step, ey_loss))

                    if np.mod(step, 50) == 0:
                        # sample_images = sess.run(self.fake_images, feed_dict={self.e_y:})
                        # save_images(sample_images[0:64], [8, 8],
                        #             './{}/train_{:02d}_{:04d}.png'.format(self.sample_path, e, step))
                        self.saver_y.save(sess, self.encode_y_model)

                e += 1
                batch_num = 0

            save_path = self.saver_y.save(sess, self.encode_y_model)
            print "Encode Y Model saved in file: %s" % save_path

    #do test
    def test(self):

        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init)

            self.saver_z.restore(sess, self.encode_z_model)
            self.saver_y.restore(sess, self.encode_y_model)

            realbatch_array, _ = MnistData.getNextBatch(self.ds_train, self.label_y, 0, 50,
                                                        self.batch_size)

            output_image , label_y = sess.run([self.fake_images,self.e_y], feed_dict={self.images: realbatch_array})

            #one-hot
            #label_y = tf.arg_max(label_y, 1)

            print label_y

            save_images(output_image , [8 , 8] , './{}/test{:02d}_{:04d}.png'.format(self.sample_path , 0, 0))
            save_images(realbatch_array , [8 , 8] , './{}/test{:02d}_{:04d}_r.png'.format(self.sample_path , 0, 0))

            gen_img = cv2.imread('./{}/test{:02d}_{:04d}.png'.format(self.sample_path , 0, 0), 0)
            real_img = cv2.imread('./{}/test{:02d}_{:04d}_r.png'.format(self.sample_path , 0, 0), 0)


            cv2.imshow("test_EGan", gen_img)
            cv2.imshow("Real_Image", real_img)

            cv2.waitKey(-1)

            print("Test finish!")

    def discriminate(self, x_var, y, weights, biases, reuse=False):

        y1 =  tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])
        x_var = conv_cond_concat(x_var, y1)

        conv1= lrelu(conv2d(x_var, weights['wc1'], biases['bc1']))

        conv1 = conv_cond_concat(conv1, y1)

        conv2= lrelu(batch_normal(conv2d(conv1, weights['wc2'], biases['bc2']), scope='dis_bn1', reuse=reuse))

        conv2 = tf.reshape(conv2, [self.batch_size, -1])

        conv2 = tf.concat([conv2, y], 1)

        fc1 = lrelu(batch_normal(fully_connect(conv2, weights['wc3'], biases['bc3']), scope='dis_bn2', reuse=reuse))

        fc1 = tf.concat([fc1, y], 1)
        #for D
        output= fully_connect(fc1, weights['wd'], biases['bd'])

        return tf.nn.sigmoid(output)

    def encode_z(self, x, weights, biases):

        c1 = tf.nn.relu(batch_normal(conv2d(x, weights['e1'], biases['eb1']), scope='enz_bn1'))

        c2 = tf.nn.relu(batch_normal(conv2d(c1, weights['e2'], biases['eb2']), scope='enz_bn2'))

        c2 = tf.reshape(c2, [self.batch_size, 128*7*7])

        #using tanh instead of tf.nn.relu.
        result_z = batch_normal(fully_connect(c2, weights['e3'], biases['eb3']), scope='enz_bn3')

        #result_c = tf.nn.sigmoid(fully_connect(c2, weights['e4'], biases['eb4']))

        #Transforming one-hot form
        #sparse_label = tf.arg_max(result_c, 1)

        #y_vec = tf.one_hot(sparse_label, 10)

        return result_z

    def encode_y(self, x, weights, biases):

        c1 = tf.nn.relu(batch_normal(conv2d(x, weights['e1'], biases['eb1']), scope='eny_bn1'))

        c2 = tf.nn.relu(batch_normal(conv2d(c1, weights['e2'], biases['eb2']), scope='eny_bn2'))

        c2 = tf.reshape(c2, [self.batch_size, 128 * 7 * 7])

        result_y = tf.nn.sigmoid(fully_connect(c2, weights['e3'], biases['eb3']))

        #y_vec = tf.one_hot(tf.arg_max(result_y, 1), 10)

        return result_y

    def generate(self, z_var, y, weights, biases):

        #add the first layer

        z_var = tf.concat([z_var, y], 1)

        d1 = tf.nn.relu(batch_normal(fully_connect(z_var , weights['wd'], biases['bd']) , scope='gen_bn1'))

        #add the second layer

        d1 = tf.concat([d1, y], 1)

        d2 = tf.nn.relu(batch_normal(fully_connect(d1 , weights['wc1'], biases['bc1']) , scope='gen_bn2'))

        d2 = tf.reshape(d2 , [self.batch_size , 7 , 7 , 128])
        y = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])

        d2 = conv_cond_concat(d2, y)

        d3 = tf.nn.relu(batch_normal(de_conv(d2, weights['wc2'], biases['bc2'], out_shape=[self.batch_size, 14 , 14 , 64]) , scope='gen_bn3'))

        d3 = conv_cond_concat(d3, y)

        output = de_conv(d3, weights['wc3'], biases['bc3'], out_shape=[self.batch_size, 28, 28, 1])

        return tf.nn.sigmoid(output)


    def get_dis_variables(self):

        weights = {

            'wc1': tf.Variable(tf.random_normal([4, 4, 1 + self.y_dim, 64], stddev=0.02), name='dis_w1'),
            'wc2': tf.Variable(tf.random_normal([4, 4, 64 + self.y_dim, 128], stddev=0.02), name='dis_w2'),
            'wc3': tf.Variable(tf.random_normal([128 * 7 * 7 + self.y_dim, 1024], stddev=0.02), name='dis_w3'),
            'wd': tf.Variable(tf.random_normal([1024 + self.y_dim, 1], stddev=0.02), name='dis_w4')
        }

        biases = {

            'bc1': tf.Variable(tf.zeros([64]), name='dis_b1'),
            'bc2': tf.Variable(tf.zeros([128]), name='dis_b2'),
            'bc3': tf.Variable(tf.zeros([1024]), name='dis_b3'),
            'bd': tf.Variable(tf.zeros([1]), name='dis_b4')
        }

        return weights, biases

    def get_en_z_variables(self):

        weights = {

            'e1': tf.Variable(tf.random_normal([4, 4, 1, 64], stddev=0.02), name='enz_w1'),
            'e2': tf.Variable(tf.random_normal([4, 4, 64, 128], stddev=0.02), name='enz_w2'),
             ##z
            'e3': tf.Variable(tf.random_normal([128 * 7 * 7, 64], stddev=0.02), name='enz_w3')
        }

        biases = {

            'eb1': tf.Variable(tf.zeros([64]), name='enz_b1'),
            'eb2': tf.Variable(tf.zeros([128]), name='enz_b2'),
             ##z
            'eb3': tf.Variable(tf.zeros([64]), name='enz_b3')
        }

        return weights, biases

    def get_en_y_variables(self):

        weights = {

            'e1': tf.Variable(tf.random_normal([4, 4, 1, 64], stddev=0.02), name='eny_w1'),
            'e2': tf.Variable(tf.random_normal([4, 4, 64, 128], stddev=0.02), name='eny_w2'),
            'e3': tf.Variable(tf.random_normal([128 * 7 * 7, 10], stddev=0.02), name='eny_w4')
        }

        biases = {

            'eb1': tf.Variable(tf.zeros([64]), name='eny_b1'),
            'eb2': tf.Variable(tf.zeros([128]), name='eny_b2'),
            'eb3': tf.Variable(tf.zeros([10]), name='eny_b4')
        }

        return weights, biases

    def get_gen_variables(self):

        weights = {

            'wd': tf.Variable(tf.random_normal([self.sample_size+self.y_dim , 1024], stddev=0.02), name='gen_w1'),
            'wc1': tf.Variable(tf.random_normal([1024 + self.y_dim , 7 * 7 * 128], stddev=0.02), name='gen_w2'),
            'wc2': tf.Variable(tf.random_normal([4, 4, 64, 128 + self.y_dim], stddev=0.02), name='gen_w3'),
            'wc3': tf.Variable(tf.random_normal([4, 4, 1, 64 + self.y_dim], stddev=0.02), name='gen_w4'),
        }

        biases = {

            'bd': tf.Variable(tf.zeros([1024]), name='gen_b1'),
            'bc1': tf.Variable(tf.zeros([7 * 7 * 128]), name='gen_b2'),
            'bc2': tf.Variable(tf.zeros([64]), name='gen_b3'),
            'bc3': tf.Variable(tf.zeros([1]), name='gen_b4')
        }

        return weights, biases



