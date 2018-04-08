# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = 2018 / 3 / 19
__version__ = ''

import networks.PA_LSTM.PartAwareLSTMCell as PartAwareLSTMCell
import tensorflow as tf
import os
import pandas as pd
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

g_feat_dim = 25*3
g_T_size = 8
g_num_action = 40


class PLSTM(object):
  _linear_layer_counter = 0
  _reg_scope = 'reg_col'
  _reg_ratio = 0.01

  def __init__(self,
               batch_size=32,
               lr=0.001,
               mode='train',
               n_layers=3,
               time_step=g_T_size):
    self._batch_size = batch_size
    self._lr = lr
    self._time_step = time_step
    self._n_layers = n_layers
    self._mode = mode

    with tf.variable_scope('net'):
      self.skel_input = tf.placeholder(dtype=tf.float32, shape=(batch_size, g_T_size, g_feat_dim), name='skel_input')
      self.action_gt = tf.placeholder(dtype=tf.float32, shape=(batch_size, g_num_action), name='skel_action_gt')
      self.plstm_keep_prob = tf.placeholder(dtype=tf.float32)

      self._build_graph()


    action_reg_loss_s = tf.summary.scalar(name='action_reg_loss', tensor=self.action_reg_loss)
    action_raw_loss_s = tf.summary.scalar(name='action_raw_loss', tensor=self.action_raw_loss)
    action_pred_acc_s = tf.summary.scalar(name='action_pred_acc', tensor=self.action_pred_acc)

    loss_summary_list = [action_reg_loss_s, action_raw_loss_s, action_pred_acc_s]
    self.loss_merged_op = tf.summary.merge(loss_summary_list)
    # tran summary

    pred_summary_list = [action_pred_acc_s]
    self.pred_mergerd_op = tf.summary.merge(pred_summary_list)


  def _build_graph(self):

    def _get_PLSTM_cells(hidden_layer):
      return PartAwareLSTMCell.PartAwareLSTMCell(num_units=hidden_layer)

    cells_list = []
    hidden_shape = int(self.skel_input.get_shape()[2])
    for num in range(self._n_layers):
      cells_list.append(_get_PLSTM_cells(hidden_layer=hidden_shape))
    for ind_lstm, each_lstm in enumerate(cells_list):
      cells_list[ind_lstm] = tf.contrib.rnn.DropoutWrapper(each_lstm, output_keep_prob=self.plstm_keep_prob)
    cells = tf.contrib.rnn.MultiRNNCell(cells_list)
    cells_state = cells.zero_state(self._batch_size, tf.float32)
    encoder_outputs, fin_state = tf.nn.dynamic_rnn(cells, self.skel_input, initial_state=cells_state)
    output_final = fin_state[-1].h

    with tf.variable_scope('fcs'):
      output_final = self._linear_connect(output_final, output_s=75, activation=tf.nn.relu)
      output_logit = self._linear_connect(output_final, output_s=g_num_action, activation=None)

    self.action_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output_logit,
                                                                        labels=self.action_gt)
    self.action_raw_loss = tf.reduce_mean(self.action_cross_entropy, name='action_raw_loss')
    tf.add_to_collection(self._reg_scope, self.action_raw_loss)
    self.action_reg_loss = tf.add_n(tf.get_collection(self._reg_scope), name='action_reg_loss')

    adam_optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    raw_grad = tf.gradients(self.action_reg_loss, train_vars)
    clip_grad, _ = tf.clip_by_global_norm(raw_grad, 10)
    self.adam_train_op = adam_optimizer.apply_gradients(zip(clip_grad, train_vars))


    self.action_distribution = tf.nn.softmax(logits=output_logit)
    self.action_argmax_target = tf.argmax(self.action_distribution, axis=1)
    self.action_predict_label = tf.one_hot(self.action_argmax_target, on_value=1.0, off_value=0.0, depth=g_num_action, name='predict_label')
    self.action_pred_acc = tf.div(x=tf.reduce_sum(self.action_predict_label*self.action_gt),
                                  y=self._batch_size)


  def _linear_connect(self, input_v, output_s, activation, is_reg=True):
    self._linear_layer_counter += 1
    weight_name = 'fc_w_%d' % self._linear_layer_counter
    bias_name = 'fc_b_%d' % self._linear_layer_counter
    input_shape = int(input_v.shape[1])

    weight = tf.get_variable(weight_name, shape=(input_shape, output_s), dtype=tf.float32)
    bias = tf.get_variable(bias_name, shape=(output_s), dtype=tf.float32, initializer=tf.zeros_initializer())

    # regularization collection
    if is_reg:
      tf.add_to_collection(self._reg_scope, tf.contrib.layers.l2_regularizer(self._reg_ratio)(weight))

    logits = tf.matmul(input_v, weight)+bias
    if activation is None:
      return logits
    else:
      return activation(logits)

  def generate_report(self, data, path):
    data_df = pd.DataFrame(data)
    writer = pd.ExcelWriter(path)
    data_df.to_excel(writer,'page_1',float_format='%.5f') # float_format 控制精度
    data_df.to_excel(writer,'page_2',float_format='%.5f') # float_format 控制精度
    writer.save()

if __name__ == '__main__':
  model = PLSTM()