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

cs_index = [291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334]
#print(type(cs_index[0]))

for i in range(0,len(dir_list)):
	sub_path = os.path.join(rgb_frames_path,dir_list[i])
	#print(sub_path)
	sub_select = int(sub_path.split('-')[0][-3:])
	action_number = int(sub_path.split('-')[-1].split('_')[1][1:])
	#print(sub_path)
	#print(sub_select)
	#print(action_number)
	sub_dir_list = os.listdir(sub_path)
	sub_dir_list.sort()
	#print(sub_dir_list)
	
	if sub_select in cs_index:
		if len(sub_dir_list) < 6:
			continue
		choice_frame = random.randint(0, len(sub_dir_list)-5)
		#print(sub_path)
		if choice_frame < 6:
			choice_frame = 5
		item_list = sub_path + ' ' + str(choice_frame) + ' ' + str(action_number - 1) + '\n'
		item_lists_test.append(item_list)
	else:
		if len(sub_dir_list) < 6:
			continue
		choice_frame = random.randint(0, len(sub_dir_list)-5)
		#print(sub_path)
		if choice_frame < 6:
			choice_frame = 5
		item_list = sub_path + ' ' + str(choice_frame) + ' '+ str(action_number - 1) +'\n'
		item_lists_train.append(item_list)
		

random.shuffle(item_lists_train)
random.shuffle(item_lists_test)

open(os.path.join(rgb_train_test_path, 'pku_cs_list_rgb_train.txt'), 'w').writelines(item_lists_train)
open(os.path.join(rgb_train_test_path, 'pku_cs_list_rgb_test.txt'), 'w').writelines(item_lists_test)

