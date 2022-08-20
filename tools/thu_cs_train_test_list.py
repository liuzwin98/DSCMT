import sys
import os
import numpy as np
import random

# official classification .txt file can be found at http://ivg.au.tsinghua.edu.cn/dataset/THU_READ.php

# ======= for rgb&depth train_test_files =============
# origin_file_path = "/home/liulb/liuz/thu_read/train_test_split/cs_4_train.txt"
# save_file_path = "/home/liulb/liuz/thu_read/thu_cs_4_train.txt"

origin_file_path = "/home/liulb/liuz/thu_read/train_test_split/cs_1_test.txt"
save_file_path = "/home/liulb/liuz/thu_read/thu_cs_1_test.txt"

origin_file = open(origin_file_path, 'r')
save_file = []
for line in origin_file.readlines():
    tmp1 = line.strip().split(' ')
    tmp1[0] = "/home/liulb/liuz/thu_read/depth/" + tmp1[0]
    save_file.append(' '.join(tmp1) + '\n')


# ======= for rp train_test_files =============
# origin_file_path = "/home/liulb/liuz/thu_read/train_test_split/cs_4_train.txt"   # train
# save_file_path = "/home/liulb/liuz/thu_read/thu_cs_4_train_rp.txt"
#
# # origin_file_path = "/home/liulb/liuz/thu_read/train_test_split/cs_4_test.txt"  # test
# # save_file_path = "/home/liulb/liuz/thu_read/thu_cs_4_test_rp.txt"
#
# origin_file = open(origin_file_path, 'r')
# save_file = []
# for line in origin_file.readlines():
#     tmp1 = line.strip().split(' ')
#     tmp1[0] = "/home/liulb/liuz/thu_read/rgb_d_rp_1/" + tmp1[0]
#     tmp1[1] = '1'   # for rp1
#     save_file.append(' '.join(tmp1) + '\n')


print(save_file)
print(len(save_file))

open(save_file_path, 'w').writelines(save_file)

