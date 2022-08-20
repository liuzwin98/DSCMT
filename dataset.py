import torch.utils.data as data
import os
import os.path
from numpy.random import randint
from transforms import *


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])  # -2 for rgb+flow or depth+flow

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='Appearance',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path  # 项目的根目录地址，如果其他文件地址使用绝对地址，则可以写成" "
        self.list_file = list_file  # 训练或测试的列表文件(.txt文件)地址
        self.num_segments = num_segments  # 视频分割的段数
        self.new_length = new_length  # 根据输入数据集类型的不同，new_length取不同的值
        self.modality = modality
        self.transform = transform
        self.random_shift = random_shift  # 若为False，则用_get_val_indices得到索引
        self.test_mode = test_mode
        self.image_tmpl = image_tmpl
        self.rp_num = 1     # 更改rp数量

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'Appearance':
            video_file = directory.split('/')[-1]

            rgb_path = '/home/liulb/liuz/ntu_rgb_frames/' + video_file
            depth_path = '/home/liulb/liuz/ntu_depth_frames/' + video_file
            # flow_path = '/home/liulb/liuz/ntu_flow/' + video_file

            img_r = Image.open(os.path.join(rgb_path, 'img_{:05d}.jpg'.format(idx))).convert('RGB')
            img_d = Image.open(os.path.join(depth_path, 'd_img_{:05d}.jpg'.format(idx))).convert('RGB')
            # img_of = Image.open(os.path.join(flow_path, 'flow_{:05d}.png'.format(idx))).convert('RGB')
            return [img_r, img_d]     # [img_r, img_of]  or  [img_d, img_of]
        elif self.modality == 'Motion':
            img1 = Image.open(os.path.join(directory, 'vdi_{:03d}.jpg'.format(idx))).convert('RGB')
            img2 = Image.open(os.path.join(directory, 'ddi_{:03d}.jpg'.format(idx))).convert('RGB')
            return [img1, img2]
        elif self.modality == 'RGB':
            directory2 = directory.split('/')
            directory2[2] = 'cq'
            directory2[3] = 'ntu_rgb_frames'
            directory2 = '/'.join(directory2)
            return [Image.open(os.path.join(directory2, 'img_{:05d}.jpg'.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        if self.modality == 'Motion':
            if self.rp_num == 3:
                average_duration = 1
            elif self.rp_num == 25:
                average_duration = (np.random.randint(3, 26) - self.new_length + 1) // self.num_segments
            else:
                average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        else:
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments

        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):
        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
        if self.modality == 'Motion' and self.rp_num == 3:
            offsets = np.array([int(x) for x in range(self.num_segments)])       # for rp3_seg3 test
        else:
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])  # for appearance
        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        if len(segment_indices) == 0:
            print(record.path, record.num_frames)
            raise ValueError('segment_idx is null!')
        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)  # origin
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)


if __name__ == '__main__':
    path = '/data/liuzhen/train_test_files/ntu120_sub_rp_test_list.txt'
    modal = 'Motion'
    # modal = 'Appearance'
    # path = '/data/liuzhen/train_test_files/ntu120_sub_depth_test_list.txt'
    loader = torch.utils.data.DataLoader(
        TSNDataSet("", path, num_segments=3,
                   new_length=1,
                   modality=modal,
                   image_tmpl="img_{:05d}.jpg",
                   transform=None
                   ),
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)
    for i, data in enumerate(loader):
        dd, lab = data
        # dd = Variable(dd)
        # lab = Variable(lab)
        print(i, "inputs:", len(dd), 'labels', lab.size())
