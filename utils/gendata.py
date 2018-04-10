# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = 2018 / 3 / 22
__version__ = ''

'''
test and train dataset generator based on x-sub or x-view
generated data formation:
Array [Number_clips, {'mat':data, 'view': v, 'class':c, 'actor':a}]
data [Frames, 25, 3]
'''

import numpy as np
import scipy.io as sio

x_mode = 'x-view'
mode = 'train'
# only consider the x-view not loop
total_views = [0, 1, 2, 3, 4, 5, 6, 7] # 0 is front view and 1-7 is 45d ti -45d, 8 is loop samples, not exist in this list
train_views_config = [7]  # modify this list to select the training views
test_views_config = [i for i in total_views if i not in train_views_config]

total_actors = range(1, 119)  # actor id begin with 1, the total actor is 118 persons
# train_actors_config = range(1, 51)
train_actors_config = np.array([  1,   2,   6,  12,  13,  16,  21,  24,  28,  29,  30,  31,  33,
        35,  39,  41,  42,  45,  47,  50,  52,  54,  55,  57,  59,  61,
        63,  64,  67,  69,  70,  71,  73,  77,  81,  84,  86,  87,  88,
        90,  91,  93,  96,  99, 102, 103, 104, 107, 108, 112, 113])
test_actors_config = [i for i in total_actors if i not in train_actors_config]

cameras_list = (1, 2)
num_actions = 40
num_camera = 2
num_views = len(total_views)
num_actors = len(total_actors)

#### config above #####

def _get_file_name(action_id, camera_id, view_id, actor_id):
  # need to modify the file name formation
  # return path+'a{}_c{}_d{}_p{:0>2d}_choose_skelton.mat'.format(action_id, camera_id, view_id, actor_id)
  return 'a{:0>2}_d{}_p{:0>3d}_c{}.mat'.format(action_id, view_id, actor_id, camera_id)


def _check_view(camera_id, view_id):
  # clips in camera 1 is all front view i.e. 0 view
  # but i eliminate all loop samples no matter it is in #1 or #2 host
  # loop sample will process individually
  if view_id == 7:
    return 'loop'
  if camera_id == 1:
    return 0
  elif camera_id == 2:
    return view_id+1
  else:
    raise ValueError()

def _check_mode(view_id, actor_id, x_mode, mode):
  if x_mode == 'x-view':
    if mode == 'train':
      return (view_id in train_views_config)
    elif mode == 'test':
      return (view_id in test_views_config)
    else:
      raise ValueError()

  elif x_mode == 'x-sub':
    if mode == 'train':
      return (actor_id in train_actors_config)
    elif mode == 'test':
      return (actor_id in test_actors_config)
    else:
      raise ValueError()

  else:
    raise ValueError()


def _read_data(file_path):
  # [frames, 25, 3]
  mat = sio.loadmat(file_path)
  mat = mat['skel']
  mat = mat.astype(dtype=np.float32)
  return mat


def _x_view_gendata(raw_path, x_mode, mode):
  datasets = []
  counter = 0
  missing_list = []

  for each_action in range(num_actions):
    for each_camera in cameras_list:
      for each_view in total_views: # (0, 7)
        for each_actor in total_actors:
          view = _check_view(view_id=each_view, camera_id=each_camera) # 1 - 8
          if view == 'loop':
            continue
          # ignore loop samples, leave (1, 7)
          if not _check_mode(view_id=view, actor_id=each_actor, x_mode=x_mode, mode=mode):
            continue
          file_name = raw_path+_get_file_name(each_action, each_camera, each_view+1, each_actor)
          try:
            data_clip = _read_data(file_name) # return x,y,z
          except FileNotFoundError:
            # print('file not exist %s' % file_name)
            missing_list.append(file_name)
            continue
          block = []
          if view in (0, 1, 2):
            block += [0]
          if view in (2, 3, 4):
            block += [1]
          if view in (4, 5, 6):
            block += [2]
          if view in (6, 7, 0):
            block += [3]

          data_dict = {
            'id':      counter, # start from 0
            'mat':     data_clip,
            'view':    view, # from 0 to 7
            'block':   block, # block id, divide 8 views into 4 blocks
            'action':  each_action, # start from 0
            'actor':   each_actor # start from 0
          }
          datasets.append(data_dict)
          counter += 1
  print(counter)
  datasets = np.array(datasets)
  return datasets

def _x_sub_gendata(raw_path, x_mode, mode):
  datasets = []
  counter = 0
  for each_action in range(num_actions):
    for each_camera in cameras_list:
      for each_view in total_views: # (0, 7)
        if mode == 'train':
          actor_list = train_actors_config
        elif mode == 'test':
          actor_list = test_actors_config
        # select valid actors

        for each_actor in actor_list:
          view = _check_view(view_id=each_view, camera_id=each_camera) # (0, 7) here each_view need to be (0 - 7)
          if view == 'loop':
            continue
          # ignore the loop samples
          file_name = raw_path+_get_file_name(each_action, each_camera, each_view+1, each_actor)
          try:
            data_clip = _read_data(file_name) # return x,y,z
          except FileNotFoundError:
            continue
          if data_clip is None:
            raise ValueError()

          block = []
          if view in (0, 1, 2):
            block += [0]
          if view in (2, 3, 4):
            block += [1]
          if view in (4, 5, 6):
            block += [2]
          if view in (6, 7, 0):
            block += [3]

          data_dict = {
            'id':      counter, # start from 0
            'mat':     data_clip,
            'view':    view, # from 0 to 7
            'block':   block, # block id, divide 8 views into 4 blocks
            'action':  each_action, # start from 0
            'actor':   each_actor # start from 0
          }
          datasets.append(data_dict)
          counter += 1
  print(counter)
  datasets = np.array(datasets)
  return datasets



def gendata(raw_path, x_mode='x-view', mode='train'):
  ## this gendata is only used on DeepLSTM,ResTCN,SkelCNN
  ## and not used on ST-GCN !
  if x_mode == 'x-view':
    dataset = _x_view_gendata(raw_path, x_mode, mode)
  elif x_mode == 'x-sub':
    dataset = _x_sub_gendata(raw_path, x_mode, mode)
  else:
    raise ValueError()
  return dataset



raw_path = '/home/fesian/AI_workspace/datasets/HRI40/raw/mats/'
#
view_save_path = '/home/fesian/AI_workspace/datasets/HRI40/hri40_new_skel/x_view/'
sub_save_path = '/home/fesian/AI_workspace/datasets/HRI40/hri40_new_skel/x_sub/'

is_view = True

if is_view:
  dat = gendata(raw_path, mode='test', x_mode='x-view')
  np.save(view_save_path+'/x_view_test_v7.npy', dat)

  dat = gendata(raw_path, mode='train', x_mode='x-view')
  np.save(view_save_path+'/x_view_train_v7.npy', dat)
else:
  dat = gendata(raw_path, mode='test', x_mode='x-sub')
  np.save(sub_save_path+'/x_sub_test.npy', dat)

  dat = gendata(raw_path, mode='train', x_mode='x-sub')
  np.save(sub_save_path+'/x_sub_train.npy', dat)


