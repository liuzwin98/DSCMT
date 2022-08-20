import sys
import cv2
import os
import numpy as np
import random

rgb_frames_path = "/home/liulb/liuz/pku_mmd/rgb/"
rgb_train_test_path = "/home/liulb/liuz/train_test_files/"

dir_list = os.listdir(rgb_frames_path) 
dir_list.sort()

item_lists_train = []
item_lists_test = []

#cs_index = [0, 1, 2, 3, 4, 5, 6, 7]
#print(type(cs_index[0]))

for i in range(0, len(dir_list)):
	sub_path = os.path.join(rgb_frames_path, dir_list[i])
	#print(sub_path)
	view_select = sub_path.split('-')[-1].split('_')[0]
	action_number = int(sub_path.split('-')[-1].split('_')[1][1:])
	#print(sub_path)
	#print(action_number)
	sub_dir_list = os.listdir(sub_path)
	sub_dir_list.sort()
	#print(sub_dir_list)
	
	if view_select == 'L':
		if len(sub_dir_list) < 6 :
			continue
		choice_frame = random.randint(1, len(sub_dir_list)-5)
		#print(sub_path)
		if choice_frame < 6:
			choice_frame = 5
		item_list = sub_path +' '+ str(choice_frame) + ' '+ str(action_number - 1) +'\n'
		item_lists_test.append(item_list)
	else:
		if len(sub_dir_list) < 6 :
			continue
		choice_frame = random.randint(1, len(sub_dir_list)-5)
		#print(sub_path)
		if choice_frame < 6:
			choice_frame = 5
		item_list = sub_path +' '+ str(choice_frame) + ' '+ str(action_number - 1) +'\n'
		item_lists_train.append(item_list)
		

random.shuffle(item_lists_train)
random.shuffle(item_lists_test)

open(os.path.join(rgb_train_test_path, 'pku_cv_list_rgb_train.txt'), 'w').writelines(item_lists_train)
open(os.path.join(rgb_train_test_path, 'pku_cv_list_rgb_test.txt'), 'w').writelines(item_lists_test)

