import argparse
import sys
import numpy as np
from utils.video_funcs import default_aggregation_func
from utils.metrics import mean_class_accuracy, softmax
from sklearn.metrics import confusion_matrix
sys.path.append('..')

parser = argparse.ArgumentParser()
# parser.add_argument('score_files', nargs='+', type=str)
parser.add_argument('--score_weights', nargs='+', type=float, default=None)
parser.add_argument('--crop_agg', type=str, choices=['max', 'mean'], default='mean')
args = parser.parse_args()

######### modify here #############
score_files = ['./score/ntu_rgb.npz', './score/ntu_depth.npz']
score_npz_files = [np.load(x, allow_pickle=True) for x in score_files]
# score_npz_files = [np.load(x, allow_pickle=True) for x in args.score_files]

if args.score_weights is None:
    score_weights = [1] * len(score_npz_files)      # 如果未指定融合权重，则为1
else:
    score_weights = args.score_weights
    if len(score_weights) != len(score_npz_files):
        raise ValueError("Only {} weight specifed for a total of {} score files"
                         .format(len(score_weights), len(score_npz_files)))

score_list = [x['scores'] for x in score_npz_files]       # x['scores'][:, 0]
label_list = [x['labels'] for x in score_npz_files]

# label verification

# score_aggregation
score_weights = [8, 1]
agg_score_list = []
normalize = False       # softmax operation
for score_vec in score_list:
    agg_score_vec = [np.mean(x, axis=0) for x in score_vec]

    if normalize:
        tmp_exp = [np.exp(x) for x in agg_score_vec]
        agg_score_vec = [x / x.sum() for x in tmp_exp]

    agg_score_list.append(np.array(agg_score_vec))

# avg融合
avg_scores = score_weights[0] * agg_score_list[0] + score_weights[1] * agg_score_list[1]

# 相乘融合
multi_scores = agg_score_list[0] * agg_score_list[1]

# max融合
max_scores = np.maximum(agg_score_list[0], agg_score_list[1])

# accuracy
avg_acc = mean_class_accuracy(avg_scores, label_list[0])
multi_acc = mean_class_accuracy(multi_scores, label_list[0])
max_acc = mean_class_accuracy(max_scores, label_list[0])
print('Avg fusion accuracy {:02f}%\nMultiply fusion accuracy {:02f}%'.format(avg_acc * 100, multi_acc * 100))
print('Max fusion accuracy {:02f}%'.format(avg_acc * 100))


video_pred_rgb = [np.argmax(np.mean(x, axis=0)) for x in score_list[0]]
video_pred_depth = [np.argmax(np.mean(x, axis=0)) for x in score_list[1]]

"""
    cf(混淆矩阵)每行表示真实类别，每列表示预测类别。因此cls_cnt = cf.sum(axis=1)表示每个真实类别有多少个video，
    cls_hit = np.diag(cf)就是将cf的对角线数据取出，表示每个类别的video中各预测对了多少个，
    因此cls_acc = cls_hit / cls_cnt就是每个类别的video预测准确率。
"""
cf_rgb = confusion_matrix(label_list[0], video_pred_rgb).astype(float)
cf_depth = confusion_matrix(label_list[0], video_pred_depth).astype(float)

cls_cnt_rgb = cf_rgb.sum(axis=1)
cls_hit_rgb = np.diag(cf_rgb)

cls_cnt_depth = cf_depth.sum(axis=1)
cls_hit_depth = np.diag(cf_depth)

cls_acc_rgb = cls_hit_rgb / cls_cnt_rgb
cls_acc_depth = cls_hit_depth / cls_cnt_depth

print("***************************************************************************************************")
print('RGB accuracy {:.02f}%'.format(np.mean(cls_acc_rgb) * 100))
print('Depth accuracy {:.02f}%'.format(np.mean(cls_acc_depth) * 100))
print("***************************************************************************************************")
