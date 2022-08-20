import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
import os
from sklearn.metrics import confusion_matrix

from dataset import TSNDataSet

from DSCMT import TSN
# from MIMN_model import TSN      # AAAI 2019
# from MMTM_model import TSN    # CVPR 2020
from transforms import *


# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['pku', 'thu', 'ntu60', 'ntu120', 'kinetics'], default='ntu120')
parser.add_argument('modality', type=str, choices=['Appearance', 'Motion'], default='Appearance')
parser.add_argument('test_list', type=str, default='/home/liulb/liuz/train_test_files/ntu120_sub_depth_test_list.txt')
# parser.add_argument('--test_list', type=str, default='/home/liuzhen/train_test_files/ntu60_cs_rp3_test_list.txt')
parser.add_argument('weights', type=str, default='./models/ntu120_cs_motion_rp1_model_best.pth.tar')
parser.add_argument('--arch', type=str, default="resnet50")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)    # train_seg_num for dynamic images
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.7)   # ren's 0.5
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=[8, 9])
parser.add_argument('--flow_prefix', type=str, default='')

args = parser.parse_args()


if args.dataset == 'ucf101':
    num_class = 101
elif args.dataset == 'kinetics':
    num_class = 400
elif args.dataset == 'ntu60':
    num_class = 60
elif args.dataset == 'ntu120':
    num_class = 120
elif args.dataset == 'sysu':
    num_class = 12
elif args.dataset == 'msr':
    num_class = 16
elif args.dataset == 'pku':
    num_class = 51
elif args.dataset == 'thu':
    num_class = 40
else:
    raise ValueError('Unknown dataset '+args.dataset)

# ========== 3 output ===========
flag = True


def eval_video(video_data, network):
    i, data, label = video_data
    num_crop = args.test_crops

    if args.modality in ['Appearance', 'Motion']:
        length = 3 * 2
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)

    with torch.no_grad():
        input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)))
        if flag:
            rst1, rst2, rst3 = network(input_var)
            rst_fus = rst3.data.cpu().numpy().copy()
        else:
            rst1, rst2 = network(input_var)

        rst_rgb = rst1.data.cpu().numpy().copy()
        rst_dep = rst2.data.cpu().numpy().copy()

    rst_rgb = rst_rgb.reshape((num_crop, args.test_segments, num_class)).mean(axis=0).reshape(
        (args.test_segments, 1, num_class))
    rst_dep = rst_dep.reshape((num_crop, args.test_segments, num_class)).mean(axis=0).reshape(
        (args.test_segments, 1, num_class))

    # TODO cheng's method 两路融合
    # lamda = 1.0
    # beta = 1.5
    # rst_add = lamda * rst_rgb + beta * rst_dep
    # rst_max = np.maximum(rst_rgb, rst_dep)
    # rst_max = 1 * rst_rgb + 3.5 * rst_dep
    # rst_max = rst_rgb * rst_dep
    # return i, rst_add, rst_max, label.numpy()[0]
    if flag:
        rst_fus = rst_fus.reshape((num_crop, args.test_segments, num_class)).mean(axis=0).reshape(
            (args.test_segments, 1, num_class))
        return rst_fus, rst_rgb, rst_dep, label.numpy()[0]  # 融合结果+原始模态结果
    else:
        return i, rst_rgb, rst_dep, label.numpy()[0]        # 原始模态结果


output_rgb = []
output_dep = []
output_fus = []
video_labels = []


def main():
    net = TSN(num_class, num_segments=1, modality=args.modality,
              base_model=args.arch,
              consensus_type=args.crop_fusion_type,
              dropout=args.dropout)

    # =============load model==================
    checkpoint = torch.load(args.weights, map_location='cpu')   # 如果没有这个参数，默认使用第0块GPU
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)
    print('==========model name: %s =============' % (args.weights.split('/')[-1]))

    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(net.input_size),
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(net.input_size, net.scale_size)
        ])
    else:
        raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

    if args.modality in ['Appearance', 'Motion']:
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    data_loader = torch.utils.data.DataLoader(
            TSNDataSet("", args.test_list, num_segments=args.test_segments,
                       new_length=data_length,
                       modality=args.modality,
                       image_tmpl=args.flow_prefix+"{}_{:05d}.jpg" if args.modality == 'Flow' else "img_{:05d}.jpg",
                       test_mode=True,
                       transform=torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=args.arch == 'BNInception'),
                           ToTorchFormatTensor(div=args.arch != 'BNInception'),
                           GroupNormalize(net.input_mean, net.input_std),
                       ])),
            batch_size=1, shuffle=False,
            num_workers=len(args.gpus) * 2 if len(args.gpus) >= 2 else 4, pin_memory=True)      # args.workers, pin_memory=True

    print('Current GPUs id:', args.gpus)
    # CPU预测
    # devices = list(range(args.workers))
    # net = torch.nn.DataParallel(net.to(devices[0]), device_ids=devices)

    # GPU预测
    torch.cuda.set_device('cuda:{}'.format(args.gpus[0]))
    net = torch.nn.DataParallel(net.cuda(), device_ids=args.gpus)
    net.eval()

    data_gen = enumerate(data_loader)
    total_num = len(data_loader.dataset)

    proc_start_time = time.time()
    max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

    for i, (data, label) in data_gen:
        if i >= max_num:
            break
        rst = eval_video((i, data, label), net)

        if flag:
            output_fus.append(rst[0])
        output_rgb.append(rst[1])
        output_dep.append(rst[2])
        video_labels.append(rst[3])
        cnt_time = time.time() - proc_start_time
        print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                        total_num,
                                                                        float(cnt_time) / (i+1)))

    video_pred_rgb = [np.argmax(np.mean(x, axis=0)) for x in output_rgb]
    video_pred_dep = [np.argmax(np.mean(x, axis=0)) for x in output_dep]

    """
        cf(混淆矩阵)每行表示真实类别，每列表示预测类别。因此cls_cnt = cf.sum(axis=1)表示每个真实类别有多少个video，
        cls_hit = np.diag(cf)就是将cf的对角线数据取出，表示每个类别的video中各预测对了多少个，
        因此cls_acc = cls_hit / cls_cnt就是每个类别的video预测准确率。
    """
    cf_rgb = confusion_matrix(video_labels, video_pred_rgb).astype(float)
    cf_dep = confusion_matrix(video_labels, video_pred_dep).astype(float)

    cls_cnt_rgb = cf_rgb.sum(axis=1)
    cls_hit_rgb = np.diag(cf_rgb)

    cls_cnt_dep = cf_dep.sum(axis=1)
    cls_hit_dep = np.diag(cf_dep)

    cls_acc_rgb = cls_hit_rgb / cls_cnt_rgb
    cls_acc_dep = cls_hit_dep / cls_cnt_dep
    if flag:
        video_pred_fus = [np.argmax(np.mean(x, axis=0)) for x in output_fus]
        cf_fus = confusion_matrix(video_labels, video_pred_fus).astype(float)
        cls_cnt_fus = cf_fus.sum(axis=1)
        cls_hit_fus = np.diag(cf_fus)
        cls_acc_fus = cls_hit_fus / cls_cnt_fus
        print('Fusion accuracy {:.02f}%'.format(np.mean(cls_acc_fus) * 100))
    print("***************************************************************************************************")
    print('RGB accuracy {:.02f}%'.format(np.mean(cls_acc_rgb) * 100))
    print('Depth accuracy {:.02f}%'.format(np.mean(cls_acc_dep) * 100))
    print("***************************************************************************************************")

    if args.save_scores is not None:

        # reorder before saving
        name_list = [x.strip().split()[0] for x in open(args.test_list)]

        order_dict = {e: i for i, e in enumerate(sorted(name_list))}

        reorder_output = [None] * len(output_rgb)
        reorder_label = [None] * len(output_rgb)
        for i in range(len(output_rgb)):
            idx = order_dict[name_list[i]]
            reorder_output[idx] = output_rgb[i]
            reorder_label[idx] = video_labels[i]

        np.savez(args.save_scores + '_rgb', scores=reorder_output, labels=reorder_label)

        # 保存第二个结果
        reorder_output_2 = [None] * len(output_dep)
        reorder_label_2 = [None] * len(output_dep)
        for i in range(len(output_dep)):
            idx = order_dict[name_list[i]]
            reorder_output_2[idx] = output_dep[i]
            reorder_label_2[idx] = video_labels[i]

        np.savez(args.save_scores + '_depth', scores=reorder_output_2, labels=reorder_label_2)

        if flag:
            reorder_output_3 = [None] * len(output_fus)
            reorder_label_3 = [None] * len(output_fus)
            for i in range(len(output_fus)):
                idx = order_dict[name_list[i]]
                reorder_output_3[idx] = output_fus[i]
                reorder_label_3[idx] = video_labels[i]

            np.savez(args.save_scores + '_fuse', scores=reorder_output_3, labels=reorder_label_3)


if __name__ == '__main__':
    main()
