import numpy as np
import os, glob
import re

import torch
import torch.utils.data
from torchvision import transforms

from PIL import Image
from pathlib import Path

frame_name_regex = re.compile(r'([0-9]+).jpg')


def frame_number(name):
    match = re.search(frame_name_regex, str(name))
    return match.group(1)


def read_video(paths, transform):
    video = []
    for path in paths:
        f = Image.open(path)
        try:
            frame = np.asarray(f, dtype=np.float32)
            if transform:
                frame = transform(frame.astype(np.uint8))
            video.append(frame)
        finally:
            if hasattr(f, 'close'):
                f.close()

    return torch.vstack([v.unsqueeze(dim=0) for v in video])


class MugDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, video_length=15, transform=None):
        self.root_path = Path(root_path)
        self.video_length = video_length
        self.extract_speed = 2
        self.transform = transform

        self.video_categories = list(self.root_path.glob("*"))
        self.num_labels = len(self.video_categories)
        self.subjects = []

        dataset_path = '/cs/cs_groups/azencot_group/datasets/MUG/subjects3/'
        # ---- create dict for one hot vector -----
        subj_dict = {}
        subject_dirs = glob.glob(os.path.join(dataset_path, '*'))
        sorted_dirs = []
        for subject_idx, subject_dir in enumerate(subject_dirs):
            sub_dirs_str = subject_dir.split('/')
            subj_idx = sub_dirs_str.index('subjects3')
            sorted_dirs.append(sub_dirs_str[subj_idx + 1])
        sorted_dirs = sorted(sorted_dirs)

        for subject_idx, sorted_dir in enumerate(sorted_dirs):
            subj_dict[sorted_dir] = subject_idx

        category2num = {
            "anger": 0,
            "disgust": 1,
            "happiness": 2,
            "fear": 3,
            "sadness": 4,
            "surprise": 5,
        }

        self.videos = []
        for category_path in self.video_categories:
            if not category_path.is_dir():
                continue

            num_categ = int(category_path.name)
            for video_path in category_path.glob("*"):
                if not video_path.is_dir():
                    continue

                video_len = len(list(video_path.glob("*.jpg")))
                if video_len >= video_length:
                    subj = 0
                    p = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'
                    if re.search(p, video_path.name) is not None:
                        for catch in re.finditer(p, video_path.name):
                            subj = catch[0]
                            break
                    self.videos.append((video_path, num_categ, subj_dict[subj]))
                else:
                    print(">> discarded {} (video length {} < {})\n".
                          format(video_path.parent.name, video_len, video_length))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i):
        """return video shape: (ch, frame, width, height)"""
        video_path, categ, subj = self.videos[i]

        frame_paths = np.array(sorted(glob.glob(os.path.join(video_path, '*.jpg')), key=frame_number))

        # videos can be of various length, we randomly sample sub-sequences
        video_len = len(frame_paths)
        if video_len < self.video_length:
            raise ValueError('invalid video length: {} < {}'
                             .format(len(frame_paths), self.video_length))
        elif video_len > self.video_length * self.extract_speed:
            needed = self.extract_speed * (self.video_length - 1)
            gap = video_len - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            subsequence_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
            frame_paths = frame_paths[subsequence_idx]
        else:
            gap = video_len - self.video_length
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            subsequence_idx = np.arange(start, start + self.video_length)
            frame_paths = frame_paths[subsequence_idx]

        # read video
        video = read_video(frame_paths, self.transform)
        if len(video.shape) != 4:
            raise ValueError('invalid video shape: {}'.format(video.shape))
        video = np.concatenate((video[:, 14:, :, :], video[:, :14, :, :]), axis=1)
        return video, categ, subj


image_transforms = transforms.Compose([
    Image.fromarray,
    # transforms.Resize(int(64)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
])
