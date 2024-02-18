import os
import time
import shutil
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import utils
from tqdm import tqdm
from dataset import Dataset
from yolov3 import YOLOV3
from config import cfg

os.environ['CUDA_VISIBLE_DEVICES'] = '1'       

class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale    = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes             = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes         = len(self.classes)
        self.num_epochs          = cfg.TRAIN.TOTAL_EPOCHS
        self.weight_dir          = cfg.TRAIN.SAVE_WEIGHT_DIR
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay    = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale = 150  # 150
        self.train_logdir        = cfg.TRAIN.TRAIN_LOG_DIR
        self.trainset            = Dataset('train')
        self.testset             = Dataset('test')
        self.steps_per_period    = len(self.trainset)

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        with tf.name_scope('define_input'):
            self.input_data   = tf.placeholder(dtype=tf.float32, name='input_data')
            self.label_sbbox  = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox  = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox  = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.trainable    = tf.placeholder(dtype=tf.bool, name='training')
            self.label_smooth = tf.placeholder(dtype=tf.bool, name='label_smooth')

        with tf.name_scope("define_loss"):
            self.model = YOLOV3(self.input_data, self.trainable, self.label_smooth)
            self.net_var = tf.global_variables()
            self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
                                                    self.label_sbbox,  self.label_mbbox,  self.label_lbbox,
                                                    self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            self.learn_rate = tf.train.cosine_decay_restarts(learning_rate=cfg.TRAIN.LEARN_RATE,
                                                             global_step=self.global_step,
                                                             first_decay_steps=self.steps_per_period*cfg.TRAIN.WARMUP_EPOCHS)
            global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope("define_single_stage_train"):
            single_stage_trainable_var_list = tf.trainable_variables()
            single_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss, var_list=single_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([single_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate", self.learn_rate)
            tf.summary.scalar("giou_loss",  self.giou_loss)
            tf.summary.scalar("conf_loss",  self.conf_loss)
            tf.summary.scalar("prob_loss",  self.prob_loss)
            tf.summary.scalar("total_loss", self.loss)

            logdir = "/home/pravin/26072020/logs/"
            if os.path.exists(logdir):
                shutil.rmtree(logdir)
            os.mkdir(logdir)
            self.write_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(logdir, graph=self.sess.graph)

    def train(self):
        module_file = "/path"
        self.sess.run(tf.global_variables_initializer())
        if module_file is not None:
            print('=> Restoring weights from ... ')
            saver = tf.train.Saver()
            saver.restore(self.sess, module_file)
        else:
            print('=> YOLO-V3 training from scratch...')

        for epoch in range(1, 1+self.num_epochs):
            train_op = self.train_op_with_all_variables

            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []
            iou_epoch_loss, conf_epoch_loss, pred_epoch_loss = [], [], []

            for train_data in pbar:
                _, summary, train_step_loss, iou_step_loss, conf_step_loss, pred_step_loss, global_step_val = self.sess.run(
                    [train_op, self.write_op, self.loss, self.giou_loss, self.conf_loss, self.prob_loss, self.global_step], feed_dict={
                                                self.input_data:   train_data[0],
                                                self.label_sbbox:  train_data[1],
                                                self.label_mbbox:  train_data[2],
                                                self.label_lbbox:  train_data[3],
                                                self.true_sbboxes: train_data[4],
                                                self.true_mbboxes: train_data[5],
                                                self.true_lbboxes: train_data[6],
                                                self.trainable:    True,
                                                self.label_smooth: True
                })

                train_epoch_loss.append(train_step_loss)
                iou_epoch_loss.append(iou_step_loss)
                conf_epoch_loss.append(conf_step_loss)
                pred_epoch_loss.append(pred_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("train loss: %.2f" % train_step_loss)

            train_epoch_loss = np.mean(train_epoch_loss)
            iou_epoch_loss, conf_epoch_loss, pred_epoch_loss = np.mean(iou_epoch_loss), np.mean(conf_epoch_loss), np.mean(pred_epoch_loss)
            with open(self.train_logdir, 'a') as f:
                f.write('Epoch: {}, Total train loss: {}\n'.format(epoch, train_epoch_loss))
                f.write('IOU loss: {},  Confidence loss: {},    Prob loss: {}\n'.format(iou_epoch_loss, conf_epoch_loss, pred_epoch_loss))

            for test_data in self.testset:
                test_step_loss = self.sess.run( self.loss, feed_dict={
                                                self.input_data:   test_data[0],
                                                self.label_sbbox:  test_data[1],
                                                self.label_mbbox:  test_data[2],
                                                self.label_lbbox:  test_data[3],
                                                self.true_sbboxes: test_data[4],
                                                self.true_mbboxes: test_data[5],
                                                self.true_lbboxes: test_data[6],
                                                self.trainable:    False,
                                                self.label_smooth: False
                })

                test_epoch_loss.append(test_step_loss)

            test_epoch_loss = np.mean(test_epoch_loss)
            ckpt_file = self.weight_dir + "yolov3_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                  % (epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            with open(self.train_logdir, 'a') as f:
                f.write('Total Test loss:{}, Log time:{}\n\n'.format(test_epoch_loss, log_time))

            if test_epoch_loss == np.isnan(test_epoch_loss):
                continue
            else:
                self.saver.save(self.sess, ckpt_file, global_step=epoch)


if __name__ == '__main__':
    YoloTrain().train()
