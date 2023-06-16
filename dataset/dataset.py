import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler, BatchSampler

from pathlib import Path
from plyfile import PlyData
import pickle, time, warnings
import numpy as np
import re

# from utils.tools import Config as cfg
from utils.tools import DataProcessing as DP

class CloudsDataset(Dataset):
    def __init__(self, dir, labeled_point, num_points, num_classes, retrain, data_type='npy'):
        self.path = dir
        self.paths = list(dir.glob(f'*.{data_type}'))
        self.size = len(self.paths)
        self.data_type = data_type
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.val_proj = []
        self.val_labels = []
        self.val_split = 'Area_6'

        ### SQN
        self.num_classes = num_classes
        self.labeled_point = labeled_point
        self.retrain = retrain
        if '%' in labeled_point:
            r = float(labeled_point[:-1]) / 100
            self.num_with_anno_per_batch = max(int(num_points * r), 1)
        else:
            self.num_with_anno_per_batch = num_classes
        self.num_per_class = np.zeros(num_classes)
        ###
        self.load_data()
        print('Size of training : ', len(self.input_colors['training']))
        print('Size of validation : ', len(self.input_colors['validation']))

    def load_data(self):
        for i, file_path in enumerate(self.paths):
            t0 = time.time()
            cloud_name = file_path.stem
            if re.match(self.val_split, cloud_name):
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = self.path / '{:s}_KDTree.pkl'.format(cloud_name)
            if self.data_type == 'npy':
                sub_file = self.path / '{:s}.npy'.format(cloud_name)
                data = np.load(sub_file, mmap_mode='r').T
            elif self.data_type == 'ply':
                sub_file = self.path / '{:s}.ply'.format(cloud_name)
                with open(sub_file, 'rb') as f:
                    plydata = PlyData.read(f)
                plydata = plydata['vertex'].data.copy()
                data = np.array(plydata)
            else:
                raise Exception("Invalid data_type!")
            sub_colors = data[:,3:6]
            sub_labels = data[:,-1].copy()
            sub_labels = sub_labels.astype(np.int32)
            
            # compute num_per_class in training set
            if cloud_split == 'training':
                self.num_per_class += DP.get_num_class_from_label(sub_labels, self.num_classes)

            # ======================================== #
            #          Random Sparse Annotation        #
            # ======================================== #
            if cloud_split == 'training':
                if '%' in self.labeled_point:
                    num_pts = len(sub_labels)
                    r = float(self.labeled_point[:-1]) / 100
                    num_with_anno = max(int(num_pts * r), 1)
                    num_without_anno = num_pts - num_with_anno
                    idx_without_anno = np.random.choice(num_pts, num_without_anno, replace=False)
                    sub_labels[idx_without_anno] = 0
                else:
                    for i in range(self.num_classes):
                        ind_per_class = np.where(sub_labels == i)[0]  # index of points belongs to a specific class
                        num_per_class = len(ind_per_class)
                        if num_per_class > 0:
                            num_with_anno = int(self.labeled_point)
                            num_without_anno = num_per_class - num_with_anno
                            idx_without_anno = np.random.choice(ind_per_class, num_without_anno, replace=False)
                            sub_labels[idx_without_anno] = 0

                # =================================================================== #
                #            retrain the model with predicted pseudo labels           #
                # =================================================================== #
                if self.retrain:
                    # TODO: retrain with pseudo labels
                    pass
                    # pseudo_label_path = './test'
                    # temp = read_ply(join(pseudo_label_path, cloud_name + '.ply'))
                    # pseudo_label = temp['pred']
                    # pseudo_label_ratio = 0.01
                    # pseudo_label[sub_labels != 0] = sub_labels[sub_labels != 0]
                    # sub_labels = pseudo_label
                    # self.num_with_anno_per_batch = int(cfg.num_points * pseudo_label_ratio)


            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            # The points information is in tree.data
            self.input_trees[cloud_split].append(search_tree)
            self.input_colors[cloud_split].append(sub_colors)
            self.input_labels[cloud_split].append(sub_labels)
            self.input_names[cloud_split].append(cloud_name)

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.name, size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices

        for i, file_path in enumerate(self.paths):
            t0 = time.time()
            cloud_name = file_path.stem

            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = self.path / '{:s}_proj.pkl'.format(cloud_name)
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)

                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    def __getitem__(self, idx):
        pass

    def __len__(self):
        # Number of clouds
        return self.size


class ActiveLearningSampler(IterableDataset):

    def __init__(self, dataset, batch_size=6, split='training', hparams=None):
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.possibility = {}
        self.min_possibility = {}
        self.hparams = hparams
        if split == 'training':
            self.n_samples = self.hparams.train_steps
        else:
            self.n_samples = self.hparams.val_steps

        #Random initialisation for weights
        self.possibility[split] = []
        self.min_possibility[split] = []
        for i, tree in enumerate(self.dataset.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

    def __iter__(self):
        return self.spatially_regular_gen()

    def __len__(self):
        return self.n_samples # not equal to the actual size of the dataset, but enable nice progress bars

    def spatially_regular_gen(self):
        # Choosing the least known point as center of a new cloud each time.

        for i in range(self.n_samples * self.batch_size):  # num_per_epoch
            # t0 = time.time()
            if self.hparams.dataset_sampling=='active_learning':
                # Generator loop

                # Choose a random cloud
                cloud_idx = int(np.argmin(self.min_possibility[self.split]))

                # choose the point with the minimum of possibility as query point
                point_ind = np.argmin(self.possibility[self.split][cloud_idx])

                # Get points from tree structure
                points = np.array(self.dataset.input_trees[self.split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=3.5 / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                if len(points) < self.hparams.num_points:
                    queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=self.hparams.num_points)[1][0]

                queried_idx = DP.shuffle_idx(queried_idx)
                # Collect points and colors
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = self.dataset.input_colors[self.split][cloud_idx][queried_idx]
                queried_pc_labels = self.dataset.input_labels[self.split][cloud_idx][queried_idx]

                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[self.split][cloud_idx][queried_idx] += delta
                self.min_possibility[self.split][cloud_idx] = float(np.min(self.possibility[self.split][cloud_idx]))

                if len(points) < self.hparams.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                        DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, self.hparams.num_points)

            # Simple random choice of cloud and points in it
            elif self.hparams.dataset_sampling=='random':

                cloud_idx = np.random.choice(len(self.min_possibility[self.split]), 1)[0]
                points = np.array(self.dataset.input_trees[self.split][cloud_idx].data, copy=False)
                queried_idx = np.random.choice(len(self.dataset.input_trees[self.split][cloud_idx].data), self.hparams.num_points)
                queried_pc_xyz = points[queried_idx]
                queried_pc_colors = self.dataset.input_colors[self.split][cloud_idx][queried_idx]
                queried_pc_labels = self.dataset.input_labels[self.split][cloud_idx][queried_idx]

                                
            if self.split == 'training':
                unique_label_value = np.unique(queried_pc_labels)
                if len(unique_label_value) <= 1:
                    continue
                else:
                    # ================================================================== #
                    #            Keep the same number of labeled points per batch        #
                    # ================================================================== #
                    idx_with_anno = np.where(queried_pc_labels != self.ignored_labels[0])[0]
                    num_with_anno = len(idx_with_anno)
                    if num_with_anno > self.dataset.num_with_anno_per_batch:
                        idx_with_anno = np.random.choice(idx_with_anno, self.dataset.num_with_anno_per_batch, replace=False)
                    elif num_with_anno < self.dataset.num_with_anno_per_batch:
                        dup_idx = np.random.choice(idx_with_anno, self.dataset.num_with_anno_per_batch - len(idx_with_anno))
                        idx_with_anno = np.concatenate([idx_with_anno, dup_idx], axis=0)
                    xyz_with_anno = queried_pc_xyz[idx_with_anno]
                    labels_with_anno = queried_pc_labels[idx_with_anno]
            else:
                xyz_with_anno = queried_pc_xyz
                labels_with_anno = queried_pc_labels

            queried_pc_xyz = torch.from_numpy(queried_pc_xyz).float()
            queried_pc_colors = torch.from_numpy(queried_pc_colors).float()
            queried_pc_labels = torch.from_numpy(queried_pc_labels).long()
            queried_idx = torch.from_numpy(queried_idx).float() # keep float here?
            cloud_idx = torch.from_numpy(np.array([cloud_idx], dtype=np.int32)).float()
            points = torch.cat( (queried_pc_xyz, queried_pc_colors), 1)

            xyz_with_anno = torch.from_numpy(xyz_with_anno).float()
            labels_with_anno = torch.from_numpy(labels_with_anno).long()

            yield points, queried_pc_labels, queried_idx, cloud_idx, xyz_with_anno, labels_with_anno


def data_loaders(dir, hparams, sampling_method='active_learning', **kwargs):
    dataset = CloudsDataset(dir, hparams.labeled_point, hparams.num_points, hparams.num_classes, hparams.retrain)
    batch_size = kwargs.get('batch_size', 6)
    hparams = hparams
    val_sampler = ActiveLearningSampler(
        dataset,
        batch_size=batch_size,
        split='validation',
        hparams=hparams
    )
    train_sampler = ActiveLearningSampler(
        dataset,
        batch_size=batch_size,
        split='training',
        hparams=hparams
    )
    return DataLoader(train_sampler, **kwargs), DataLoader(val_sampler, **kwargs)

    # if sampling_method == 'naive':
    #     train_dataset = PointCloudsDataset(dir)
    #     val_dataset = PointCloudsDataset(dir / 'val')
    #     return DataLoader(train_dataset, shuffle=True, **kwargs), DataLoader(val_dataset, **kwargs)

    # raise ValueError(f"Dataset sampling method '{sampling_method}' does not exist.")

if __name__ == '__main__':
    dataset = CloudsDataset(Path('/share/dataset/sqn_own/S3DIS/train'))
    batch_sampler = ActiveLearningSampler(dataset)
    for data in batch_sampler:
        xyz, colors, labels, idx, cloud_idx = data
        print('Number of points:', len(xyz))
        print('Point position:', xyz[1])
        print('Color:', colors[1])
        print('Label:', labels[1])
        print('Index of point:', idx[1])
        print('Cloud index:', cloud_idx)
        break
