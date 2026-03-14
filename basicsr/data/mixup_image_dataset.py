from torch.utils import data as data
from torchvision.transforms.functional import normalize
import numpy as np
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MixupDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(MixupDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        random_int = np.random.randint(low=0, high=self.__len__()-1) #random select other mixed images
        gt_1_path = self.paths[index]['gt_path']
        gt_2_path = self.paths[random_int]['gt_path']
        img_1_bytes = self.file_client.get(gt_1_path, 'gt')
        img_1_gt = imfrombytes(img_1_bytes, float32=True)
        img_2_bytes = self.file_client.get(gt_2_path, 'gt')
        img_2_gt = imfrombytes(img_2_bytes, float32=True)


        lq_1_path = self.paths[index]['lq_path']
        lq_2_path = self.paths[random_int]['lq_path']
        img_1_bytes = self.file_client.get(lq_1_path, 'lq')
        img_1_lq = imfrombytes(img_1_bytes, float32=True)
        img_2_bytes = self.file_client.get(lq_2_path, 'lq')
        img_2_lq = imfrombytes(img_2_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_1_gt, img_1_lq = paired_random_crop(img_1_gt, img_1_lq, gt_size, scale, gt_1_path)
            img_2_gt, img_2_lq = paired_random_crop(img_2_gt, img_2_lq, gt_size, scale, gt_2_path)
        #     # flip, rotation
            img_1_gt, img_1_lq = augment([img_1_gt, img_1_lq], self.opt['use_hflip'], self.opt['use_rot'])
            img_2_gt, img_2_lq = augment([img_2_gt, img_2_lq], self.opt['use_hflip'], self.opt['use_rot'])

        #Do Mixup
        alpha = np.random.uniform(self.opt['alpha_low'], self.opt['alpha_high'])
        img_gt = img_1_gt * alpha + img_2_gt * (1 - alpha)
        img_lq = img_1_lq * alpha + img_2_lq * (1 - alpha)

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq1': img_1_lq, 'lq2': img_2_lq, 'gt1': img_1_gt,
                'gt2': img_2_gt, 'lq_path': lq_1_path, 'gt_path': gt_1_path,
                'lq_2_path': lq_2_path, 'gt_2_path': gt_2_path}

    def __len__(self):
        return len(self.paths)
