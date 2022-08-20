## Dual-Stream Cross-Modality Fusion Transformer for RGB-D Action Recognition

[comment]: <> (This repo holds the code for the work on *Knowledge-Based System* )

[comment]: <> ([[Paper]]&#40;url&#41;)

The paper is accepted by the *Knowledge-Based System (KBS)*, and the link will be given after the paper is published.

# Usage Guide

## Prerequisites
The code is built with the following libraries:

- Python 3.7
- PyTorch 1.10
- opencv-python  4.5.1.48
- Pillow 8.3.1

## Data Preparation
Generate txt files of training set and test set through `./tools/dataset_protocol_train_test_list.py`


## Training
To train a new model, use the `main.py` script. The different evaluation protocols of the dataset are determined by the txt file.

```bash
# for RGB+depth raw data (cross-subject) 
python main.py ntu120 Appearance path_of_ntu120_sub_train_list.txt \
  --arch resnet50 --num_segment 1 --lr 0.001 --lr_steps 25 50 \
  --epochs 60 -b 64 --snapshot_pref saved_model_name \
  --val_list path_of_ntu120_sub_val_list.txt --gpus 0 1
```

```bash
# for RGB+depth dynamic images (cross-subject) 
python main.py ntu120 Motion path_of_ntu120_sub_rp_train_list.txt \
  --arch resnet50 --num_segment 1 --lr 0.001 --lr_steps 25 50 \
  --epochs 60 -b 64 --snapshot_pref saved_model_name \
  --val_list path_of_ntu120_sub_val_list.txt --gpus 0 1
```

For RGB+optical_flow and depth+optical_flow, please modify `_load_image()` in `dataset.py`

## Testing
After training, there will be a checkpoint file whose name contains the accuracy on the validation set and the number of epoch.

Use the following command to test its performance:

```bash
# for RGB+depth raw data (cross-subject) 
  
python test_models.py ntu120 Appearance path_of_ntu120_sub_test_list.txt \
  ntu120_sub_appearance_model_best.pth.tar --arch resnet50 --save_scores ./score/ntu120_sub_app_seg1 \
  --test_segments 1 --gpus 3 
```

```bash
# for RGB+depth dynamic images (cross-subject) 
python test_models.py ntu120 Motion path_of_ntu120_sub_rp_test_list.txt \
  ntu120_sub_motion_model_best.pth.tar --arch resnet50 --save_scores ./score/ntu120_sub_mot_seg1 \
  --test_segments 1 --gpus 3 
```


## Acknowledgements
We thank [@yjxiong][yjxiong] for sharing TSN-Pytorch codebase.

[yjxiong]: https://github.com/yjxiong/tsn-pytorch
