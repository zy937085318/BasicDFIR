# Code adapted from
#
# Tong, A., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Fatras, K., ... & Bengio, Y. (2023).
# Improving and generalizing flow-based generative models with minibatch optimal transport.
# arXiv preprint arXiv:2302.00482.
# (https://github.com/atong01/conditional-flow-matching)
#
# Chemseddine, J., Hagemann, P., Wald, C., & Steidl, G. (2024).
# Conditional Wasserstein Distances with Applications in Bayesian OT Flow Matching.
# arXiv preprint arXiv:2403.18705.
# (https://github.com/JChemseddine/Conditional_Wasserstein_Distances/blob/main/utils/utils_FID.py)

import torch
import os
import skimage.io as io
import numpy as np
import torch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import ot
from torchdiffeq import odeint_adjoint as odeint
import pnpflow.fid_score as fs
from pnpflow.dataloaders import DataLoaders
import pnpflow.utils as utils
from fld.metrics.FID import FID
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from pnpflow.dataloaders import CelebADataset, AFHQDataset
from torchvision import transforms
from torchvision.utils import save_image


img_dir_celeba = './data/celeba/img_align_celeba/'
partition_csv_celeba = './data/celeba/list_eval_partition.csv'
img_dir_afhq = '.data/afhq_cat/test/cat/'


class FLOW_MATCHING(object):

    def __init__(self, model, val_dataset, device, args):
        self.d = args.dim_image #image 尺寸
        self.num_channels = args.num_channels
        self.device = device
        self.args = args
        self.lr = args.lr
        self.model = model.to(device)
        self.val_dataset = val_dataset

    def train_FM_model(self, train_loader, val_loader, opt, num_epoch):

        # ft_extractor = InceptionFeatureExtractor(save_path="features")
        # if self.args.dataset == "celeba":
        #     test_feat = ft_extractor.get_features(self.val_dataset)
        # elif self.args.dataset == "afhq_cat":
        #     test_feat = AFHQDataset(
        #         img_dir_afhq, batchsize=self.batch_size_test, transform = transforms.Compose([transforms.Resize((256, 256)),
        #         transforms.ToTensor()]))
        # else:
        #     raise ValueError(f"Unknown dataset {self.args.dataset}")
            

        tq = tqdm(range(num_epoch), desc='epoch')
        for ep in tq:
            for iteration, (x, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc='iters'):
                if x.size(0) == 0:
                    continue
                # if iteration > 20:
                #     break
                # print(f'Epoch: {ep}, iter: {iteration}')
                x = x.to(self.device)
                lr = torch.nn.functional.interpolate(x, scale_factor=1/4, mode='bilinear', align_corners=False)
                lr = torch.nn.functional.interpolate(lr, scale_factor=4, mode='bilinear', align_corners=False)
                z = torch.randn(
                    x.shape[0],
                    3, # self.num_channels,
                    self.d,
                    self.d,
                    device=self.device,
                    requires_grad=True)
                t1 = torch.rand(x.shape[0], 1, 1, 1, device=self.device)

                # compute coupling
                x0 = z.clone()
                x1 = x.clone()
                # a, b = np.ones(len(x0)) / len(x0), np.ones(len(x0)) / len(x0)
                #
                # M = ot.dist(x0.view(len(x0), -1).cpu().data.numpy(),
                #             x1.view(len(x1), -1).cpu().data.numpy())
                # plan = ot.emd(a, b, M)
                # p = plan.flatten()
                # p = p / p.sum()
                # choices = np.random.choice(
                #     plan.shape[0] * plan.shape[1], p=p, size=len(x0), replace=True)
                # i, j = np.divmod(choices, plan.shape[1])
                # x0 = x0[i]
                # x1 = x1[j]
                xt = t1 * x1 + (1 - t1) * x0
                b = xt
                xt = torch.cat([xt, lr], dim=1)
                # if t1.squeeze().shape != 16:
                #     print(t1.squeeze().shape)
                loss = torch.sum(
                    (self.model(xt, t1.squeeze()) - (x1 - x0))**2) / x.shape[0]
                # utils.save_samples(x.detach().cpu(), x1.detach().cpu(), self.save_path + 'results_trains/' +
                #                    'trains_sr_ep_{}.pdf'.format(ep), self.args)
                opt.zero_grad()
                loss.backward()
                opt.step()

                # save loss in txt file
                with open(self.save_path + 'loss_training.txt', 'a') as file:
                    file.write(
                        f'Epoch: {ep}, iter: {iteration}, Loss: {loss.item()}\n')

            # save samples, plot them, and compute FID on small dataset
            self.sample_plot(val_loader, ep)
            # if ep % 5 == 0:
            #     # save model
            #     torch.save(self.model.state_dict(),
            #                self.model_path + 'model_{}.pt'.format(ep))
            #     # evaluate FID
            #     print("Computing FID 5K")
            #     num_gen = 5_000
                # fid_value = self.compute_fid(num_gen, test_feat,
                #                         ft_extractor, batch_size=124, integration_method="euler", integration_steps=10)
                #
                # with open(self.save_path + f'FID_{(num_gen // 1000)}k.txt', 'a') as file:
                #     file.write(f'Epoch: {ep}, FID: {fid_value}\n')

    def apply_flow_matching(self, val_loader):
        self.model.eval()
        sr_list = []
        hr_list = []
        for step, (hr, _) in enumerate(val_loader):
            lr = torch.nn.functional.interpolate(hr, scale_factor=1 / 4, mode='bilinear', align_corners=False)
            lr = torch.nn.functional.interpolate(lr, scale_factor=4, mode='bilinear', align_corners=False).to(self.device)
            with torch.no_grad():
                model_class = cnf(self.model, lr) #continuous normalize flow
                latent = torch.randn(
                    1,
                    self.num_channels//2,#和lr拼接后才是num_channel
                    self.d,
                    self.d,
                    device=self.device,
                    requires_grad=False)
                z_t = odeint(model_class, latent,
                             torch.tensor([0.0, 1.0]).to(self.device),
                             atol=1e-5,
                             rtol=1e-5,
                             method='dopri5',
                             )
                x = z_t[-1].detach()
                sr_list.append(x)
                hr_list.append(hr)
        self.model.train()
        return torch.cat(sr_list, dim=0), torch.cat(hr_list, dim=0)

    def sample_plot(self, val_loader, ep=None):
        try:
            os.makedirs(self.save_path + 'results_samplings/', exist_ok=True)
        except BaseException:
            pass

        sr, hr = self.apply_flow_matching(val_loader)
        # reco = utils.postprocess(reco, self.args)#对图像色域进行转换

        utils.save_samples(sr.detach().cpu(), hr.cpu(), self.save_path + 'results_samplings/' +
                           'samplings_sr_ep_{}.pdf'.format(ep), self.args)

        # check the plots by saving training samples
        # if ep == 0:
        #     gt = x[:16]
        #     gt = utils.postprocess(gt, self.args)
        #     utils.save_samples(gt.detach().cpu(), gt.detach().cpu(), self.save_path + 'results_samplings/' +
        #                        'train_samples_ep_{}.pdf'.format(ep), self.args)

    def generate_samples(self, integration_method="dopri5", tol=1e-5,
                         n_samples=1028, batch_size=None, num_channels=3,
                         integration_steps=100, tmax=1):
        """
        Return a tensor of size (TODO).
        """

        if batch_size is None:
            batch_size = n_samples

        images_list = []
        batches = [batch_size] * (n_samples // batch_size)
        if n_samples % batch_size:
            batches += [n_samples % batch_size]

        with torch.no_grad():
            for k, batch in enumerate(tqdm(batches)):
                time_points = torch.linspace(
                    0, tmax, int(tmax * integration_steps), device=self.device)

                x0 = torch.randn(batch, num_channels, self.d,
                                 self.d, device=self.device)
                model_class = cnf(self.model)
                traj = odeint(model_class, x0, time_points, rtol=tol, atol=tol,
                    method=integration_method)
                images_list.append(traj[-1, :])

        images = torch.cat(images_list, dim=0)
        return images
    
    def compute_fid(self, num_images_fid, train_feat, ft_extractor, batch_size=512, integration_method="dopri5", integration_steps=100,  epoch='final'):
        gen_images = self.generate_samples(integration_method=integration_method, tol=1e-4,
                                           n_samples=num_images_fid, batch_size=batch_size, integration_steps=integration_steps)
        rescaled_imgs = (gen_images * 127.5 + 128).clip(0, 255).to(torch.uint8)
        gen_feat = ft_extractor.get_tensor_features(
            rescaled_imgs)

        fid_val = FID().compute_metric(
            train_feat, None, gen_feat)

        # save the 16 first generated images in a grid
        os.makedirs(f"training_images/{self.args.dataset}", exist_ok=True)
        images = gen_images[:16]
        save_image(images, f"training_images/{self.args.dataset}/gen_images_epoch{epoch}.png")
        return fid_val

    def train(self, data_loaders):

        self.save_path = self.args.root + \
            'results_sr_jit/{}/ot/'.format(
                self.args.dataset)
        try:
            os.makedirs(self.save_path, exist_ok=True)
        except BaseException:
            pass

        self.model_path = self.args.root + \
            'model_sr/{}/ot/'.format(
                self.args.dataset)
        try:
            os.makedirs(self.model_path, exist_ok=True)
        except BaseException:
            pass

        # load model
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']

        # create txt file for storing all information about model
        with open(self.save_path + 'model_info.txt', 'w') as file:
            file.write(f'PARAMETERS\n')
            file.write(
                f'Number of parameters: {sum(p.numel() for p in self.model.parameters())}\n')
            file.write(f'Number of epochs: {self.args.num_epoch}\n')
            file.write(f'Batch size: {self.args.batch_size_train}\n')
            file.write(f'Learning rate: {self.lr}\n')

        # start training
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.train_FM_model(train_loader, val_loader, opt, num_epoch=self.args.num_epoch)

        # save final model
        torch.save(self.model.state_dict(), self.model_path + 'model_final.pt')


class cnf(torch.nn.Module):

    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, t, x):
        with torch.no_grad():
            # z = self.model(x, t.squeeze())
            z = self.model(torch.cat([x, self.lr],dim=1), t.repeat(x.shape[0]))
        return z


