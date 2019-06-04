# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2018/1/28'
__version__ = ''

import tensorflow as tf
import os
import numpy as np
import random
from keras.utils import to_categorical

g_feat_dim = 75
g_num_actions = 40

class DataLoader(object):
  __skel_train_dataset = None
  __skel_test_dataset = None
  __train_ids = None
  __valid_ids = None
  __mode = None

  def __init__(self, skel_train, skel_test, mode, lstm_tim_step=8):
    self.__mode = mode
    # divide the origin training set into train part and validation part

    if mode == 'train':
      self.__skel_train_dataset = skel_train
    elif mode == 'test':
      self.__skel_test_dataset = skel_test
    elif mode == 'both':
      self.__skel_train_dataset = skel_train
      self.__skel_test_dataset = skel_test
    else:
      self.__skel_test_dataset = None
      self.__skel_train_dataset = None

    self.lstm_time_step = lstm_tim_step

    if self.__skel_train_dataset is not None:
      self.train_index = list(range(self.__skel_train_dataset.shape[0]))
      random.shuffle(self.train_index)
      self.__skel_train_dataset = self.__skel_train_dataset[self.train_index]

  def _y_transmat(self, thetas):
    tms = np.zeros((0, 3, 3))
    thetas = thetas*np.pi/180
    for theta in thetas:
      tm = np.zeros((3, 3))
      tm[0, 0] = np.cos(theta)
      tm[0, 2] = -np.sin(theta)
      tm[1, 1] = 1
      tm[2, 0] = np.sin(theta)
      tm[2, 2] = np.cos(theta)
      tm = tm[np.newaxis, :, :]
      tms = np.concatenate((tms, tm), axis=0)
    return tms

  def _pararell_skeleton(self, raw_mat):
    '''
    raw_mat with the shape of (nframes, 25*3)
    '''
    joints_list = []

    for each_joints in range(25):
      joints_list.append(raw_mat[:, each_joints*3:each_joints*3+3])

    right_shoulder = joints_list[8] # 9th joint
    left_shoulder = joints_list[4] # 5tf joint
    vec = right_shoulder-left_shoulder
    vec[:, 1] = 0
    l2_norm = np.sqrt(np.sum(np.square(vec), axis=1))
    theta = vec[:, 0]/(l2_norm+0.0001)
    # print(l2_norm)
    thetas = np.arccos(theta)*(180/np.pi)
    isv = np.sum(vec[:, 2])
    if isv >= 0:
      thetas = -thetas
    y_tms = self._y_transmat(thetas)

    new_skel = np.zeros(shape=(0, 25*3))
    for ind, each_s in enumerate(raw_mat):
      r = np.reshape(each_s, newshape=(25, 3))
      r = np.transpose(r)
      r = np.dot(y_tms[ind], r)
      r_t = np.transpose(r)
      r_t = np.reshape(r_t, newshape=(1, -1))
      new_skel = np.concatenate((new_skel, r_t), axis=0)
    return new_skel


  def __T_clips(self, mat, T_size):
    samples_num = mat.shape[0]
    each_clip_size = int(samples_num/T_size)
    index_list = []
    begin = 0
    end = 0
    for each in range(T_size):
      end += each_clip_size
      index_list.append((begin, end))
      begin = end
    random_list = []
    for each_index in index_list:
      random_id = random.sample(list(range(each_index[0], each_index[1])), 1)[0]
      random_list.append(random_id)
    sample_mat = mat[random_list]
    return sample_mat

  def get_samples(self, batch_size, anchor, mode='train', is_rotate=True):

    mat_batch = np.zeros(shape=(batch_size, self.lstm_time_step, g_feat_dim))
    label_batch = np.zeros(shape=(batch_size, g_num_actions))
    view_batch = np.zeros(shape=(batch_size, 1))
    for each in range(batch_size):
      data = self.__skel_train_dataset[anchor+each]
      mat = data['mat']
      label = data['action']
      view = data['view']

      mat = self.__T_clips(mat, T_size=self.lstm_time_step)
      mat = np.reshape(mat, newshape=(-1, g_feat_dim))
      if is_rotate:
        mat = self._pararell_skeleton(mat)
      label = to_categorical(label, num_classes=g_num_actions)

      mat_batch[each] = mat
      label_batch[each] = label
      view_batch[each] = view
    return mat_batch, label_batch, view_batch

  def get_test_sample(self, batch_size, ind, is_rotate):
    mat_batch = np.zeros(shape=(batch_size, self.lstm_time_step, g_feat_dim))
    label_batch = np.zeros(shape=(batch_size, 1))
    view_batch = np.zeros(shape=(batch_size, 1))
    for each in range(batch_size):
      data = self.__skel_test_dataset[ind+each]
      mat = data['mat']
      label = data['action']
      view = data['clip_ind']
      mat = self.__T_clips(mat, T_size=self.lstm_time_step)
      mat = np.reshape(mat, newshape=(-1, g_feat_dim))
      if is_rotate:
        mat = self._pararell_skeleton(mat)
      mat_batch[each] = mat
      label_batch[each] = label
      view_batch[each] = view
    return mat_batch, label_batch, view_batch





if __name__ == '__main__':
  pass


