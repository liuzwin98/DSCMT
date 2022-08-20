import sys
import os
import numpy as np
import random

"""Each file/folder name in both datasets is in the format of 
SsssCcccPpppRrrrAaaa (e.g., S001C002P003R002A013), 
in which sss is the setup number, ccc is the camera ID, ppp is the performer
 (subject) ID, rrr is the replication number (1 or 2), and aaa is the action class label.

NTU60: S001-S017,  NTU120: S001-S032
 """

rgb_frames_path = "/home/liulb/liuz/ntu_depth_frames"

rgb_train_test_path = "/home/liulb/liuz/train_test_files"

dir_list = os.listdir(rgb_frames_path)

item_lists_train = []
item_lists_test = []


for i in range(0, len(dir_list)):
    sub_path = os.path.join(rgb_frames_path, dir_list[i])  # e.g., /data/liuzhen/ntu_depth_frames/S001C002P003R002A013
    setup_number = int(sub_path.split('/')[-1][2:4])
    if (setup_number % 2) == 0:   # for ntu120, even number for training
        if os.path.isdir(sub_path):
            sub_dir_list = os.listdir(sub_path)
            total_frame = len(sub_dir_list)

            # 因为类别编号是从1开始的，所以-1。 eg：S001C002P003R002A013 34 12
            item_list = sub_path + ' ' + str(total_frame-1) + ' ' + str(int(dir_list[i][-3:]) - 1) + '\n'
            item_lists_train.append(item_list)
    else:
        if os.path.isdir(sub_path):
            sub_dir_list = os.listdir(sub_path)
            total_frame = len(sub_dir_list)

            item_list = sub_path + ' ' + str(total_frame-1) + ' ' + str(int(dir_list[i][-3:]) - 1) + '\n'
            item_lists_test.append(item_list)
    print("Done %d" % (i + 1))

random.shuffle(item_lists_train)
random.shuffle(item_lists_test)
print(len(item_lists_train))
print(len(item_lists_test))

open(os.path.join(rgb_train_test_path, 'ntu120_setup_depth_train_list.txt'), 'w').writelines(item_lists_train)
open(os.path.join(rgb_train_test_path, 'ntu120_setup_depth_test_list.txt'), 'w').writelines(item_lists_test)


