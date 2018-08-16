import numpy as np

import torch
import torch.nn as nn

from sklearn import random_projection
from sklearn import decomposition

import utils

def choose_archG(args):
    arch = {'v0':DCGAN_G, 'v1':DCGAN_Gv1}
    return arch[args.archG](image_size=args.syn_size, nc=args.nc, nz=args.nz, ngf=args.ngf, n_extra_layers=args.g_extra)

def choose_archE(args):
    arch = {'pca':PCA, 'randproj':RandomProjection, 'autoencoder':Encoder}
    if args.archE == 'autoencoder':
        netE = arch[args.archE](patch_size=args.patch_size, nc=args.nc, ncode=args.ncode, ndf=args.ndf)
        netE.load_state_dict(torch.load(args.netE))
        for p in netE.parameters(): p.requires_grad_(False)
        netE.eval()
        return netE
    elif args.archE == 'pca':
        netE = arch[args.archE](patch_size=args.patch_size, nc=args.nc, n_components=args.ncode)
        netE.fit(args.img)
        return netE
    elif args.archE == 'randproj':
        return arch[args.archE](patch_size=args.patch_size, nc=args.nc, n_components=args.ncode)

def _forward(func, x):
    """ Handle superbatches """
    if len(x.shape) == 5:
        b, npatch, nc, patch_size, patch_size = x.shape
        x = x.view(b*npatch, nc, patch_size, patch_size)
        codes = func(x)  # (b*npatch, ncode, 1, 1)
        return codes.view(b, npatch, *codes.shape[1:])
    else:
        codes = func(x)
        return codes

class ConvLayer(nn.Module):
    """ Conv2d with reflection padding """
    def __init__(self, in_channels, out_channels, filter_size, stride, padding, bias=False):
        super(ConvLayer, self).__init__()
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, filter_size, stride, 0, bias=bias)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class UpsampleConvLayer(nn.Module):
    """ Upsample (nearest) + ReflectionPad + Conv2d """
    def __init__(self, in_channels, out_channels, filter_size=3, stride=1, padding=1, bias=False, upsample=2):
        super(UpsampleConvLayer, self).__init__()
        self.upsample_layer = nn.Upsample(mode='nearest', scale_factor=upsample)
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, filter_size, stride, 0, bias=bias)

    def forward(self, x):
        x = self.upsample_layer(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class UpBlock(nn.Module):
    """ UpsampleConvLayer + BN/IN + relu/leakyrelu """
    def __init__(self, in_channels, out_channels,
                 normalization='batchnorm', activation='relu'):
        super(UpBlock, self).__init__()

        main = nn.Sequential()
        main.add_module('conv', UpsampleConvLayer(in_channels, out_channels))

        if normalization == 'instancenorm':
            main.add_module('instancenorm', nn.InstanceNorm2d(out_channels, affine=True))
        elif normalization == 'batchnorm':
            main.add_module('batchnorm', nn.BatchNorm2d(out_channels))

        if activation == 'relu':
            main.add_module('relu', nn.ReLU(True))
        elif activation == 'leakyrelu':
            main.add_module('leakyrelu', nn.LeakyReLU(0.2, inplace=True))

        self.main = main

    def forward(self, x):
        return self.main(x)

class X(nn.Module):
    """ x = tanh(x') or sigmoid(x') """
    def __init__(self, image_size, nc, batch_size=1, activation='tanh'):
        super(X, self).__init__()
        self.z = nn.Parameter(torch.randn(batch_size, nc, image_size, image_size))

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('invalid activation')

    def forward(self):
        return self.activation(self.z)

class DCGAN_G(nn.Module):
    """ Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (Radford et al.) """
    def __init__(self, image_size, nc, nz, ngf, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
        assert image_size % 16 == 0, "image_size has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != image_size:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial_{0}-{1}_convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial_{0}_batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial_{0}_relu'.format(cngf),
                        nn.ReLU(True))

        csize = 4
        while csize < image_size//2:
            main.add_module('pyramid_{0}-{1}_convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid_{0}_relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}_{1}_conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}_{1}_batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}_{1}_relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final_{0}-{1}_convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final_{0}_tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, x):
        return self.main(x)

class DCGAN_Gv1(nn.Module):
    """ A version of DCGAN_G with UpsampleConvLayer instead of ConvTransposed """
    def __init__(self, image_size, nc, nz, ngf, n_extra_layers=0):
        super(DCGAN_Gv1, self).__init__()
        assert image_size % 16 == 0, "image_size has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != image_size:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial_{0}-{1}_convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial_{0}_batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial_{0}_relu'.format(cngf),
                        nn.ReLU(True))

        csize = 4
        while csize < image_size//2:
            main.add_module('pyramid_{0}-{1}_convt'.format(cngf, cngf//2),
                            UpsampleConvLayer(cngf, cngf//2, 3, 1, 1, bias=False, upsample=2))
            main.add_module('pyramid_{0}_batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid_{0}_relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}_{1}_conv'.format(t, cngf),
                            ConvLayer(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}_{1}_batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}_{1}_relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final_{0}-{1}_convt'.format(cngf, cngf),
                        UpsampleConvLayer(cngf, nc, 3, 1, 1, bias=False, upsample=2))
        main.add_module('final_{0}_tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, x):
        return self.main(x)

class DCGAN_Gv2(nn.Module):
    """ Similar to Ulyanov et. al. (supplementary material) """
    def __init__(self, image_size, nc, nz, ngf, n_extra_layers=0):
        super(DCGAN_Gv2, self).__init__()
        assert image_size % 16 == 0, "image_size has to be a multiple of 16"

        NUPSAMPLE = 3

        cngf, tisize = ngf//2, 4
        while tisize != image_size:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial_{}-{}_convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial_{}_batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial_{}_relu'.format(cngf),
                        nn.ReLU(True))

        csize = 4
        while csize < image_size//(2**NUPSAMPLE):
            main.add_module('pyramid_{}-{}_convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid_{}_batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid_{}_relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{}_{}_conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{}_{}_batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{}_{}_relu'.format(t, cngf),
                            nn.ReLU(True))

        for _ in range(1,NUPSAMPLE):
            main.add_module('upsample_{}_{}'.format(cngf, cngf//2), 
                            UpsampleConvLayer(cngf, cngf//2, 3, 1, 1, bias=False))
            main.add_module('upsample_{}_batchnorm'.format(cngf//2), 
                            nn.BatchNorm2d(cngf//2))
            main.add_module('upsample_{}_relu'.format(cngf//2), 
                            nn.ReLU(True))
            cngf = cngf // 2

        main.add_module('final_{}-{}_upsample'.format(cngf, nc),
                        UpsampleConvLayer(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final_{}_tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, x):
        return self.main(x)

class Encoder(nn.Module):
    """ Similar to DCGAN_D (Radford et al.) with tanh output and ncode """
    def __init__(self, patch_size, nc, ncode, ndf, n_extra_layers=0):
        super(Encoder, self).__init__()
        assert patch_size % 16 == 0, "image_size has to be a multiple of 16"

        main = nn.Sequential()
        main.add_module('initial_{}-{}_conv'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial_{}_batchnorm'.format(ndf),
                        nn.BatchNorm2d(ndf))
        main.add_module('initial_{}_lrelu'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = patch_size / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra_{}-{}_conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra_{}-{}_batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra_{}-{}_lrelu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            main.add_module('pyramid_{}-{}_conv'.format(cndf, cndf * 2),
                            nn.Conv2d(cndf, cndf * 2, 4, 2, 1, bias=False))
            main.add_module('pyramid_{}_batchnorm'.format(cndf * 2),
                            nn.BatchNorm2d(cndf * 2))
            main.add_module('pyramid_{}_lrelu'.format(cndf * 2),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        main.add_module('final_{}-{}_conv'.format(cndf, ncode),
                        nn.Conv2d(cndf, ncode, 4, 1, 0, bias=False))
        main.add_module('final_{}_tanh'.format(ncode),
                        nn.Tanh())
        self.main = main

    def forward(self, x):
        # return self.main(x)
        return _forward(self.main, x)

class RandomProjection(nn.Module):
    def __init__(self, patch_size, nc, n_components, stride=1):
        super(RandomProjection, self).__init__()

        randmat = random_projection.gaussian_random_matrix(n_components, nc*patch_size*patch_size)
        self.main = nn.Conv2d(nc, n_components, patch_size, stride=stride, padding=0, bias=False)
        randmat = randmat.reshape(*self.main.weight.shape).astype(np.float32)
        self.main.weight = nn.Parameter(torch.from_numpy(randmat))

        for p in self.parameters(): p.requires_grad = False

    def forward(self, x):
        # return self.main(x)
        return _forward(self.main, x)

class PCA(nn.Module):
    def __init__(self, patch_size, nc, n_components, stride=1):
        super(PCA, self).__init__()
        self.main = nn.Conv2d(nc, n_components, patch_size, stride=stride, padding=0, bias=False)

        self.patch_size = patch_size
        self.nc = nc
        self.n_components = n_components
    
    def fit(self, img):
        patches = utils.get_patches(img, self.patch_size)
        patches = patches.reshape(patches.shape[0], -1)
        main = decomposition.PCA(n_components=self.n_components)
        main.fit(patches)
        eigv = main.components_
        mean = main.mean_
        bias = -eigv.dot(mean)
        weight = eigv.reshape(self.n_components, img.shape[0], self.patch_size, self.patch_size)

        self.main.weight = nn.Parameter(torch.from_numpy(weight.astype(np.float32)))
        self.main.bias = nn.Parameter(torch.from_numpy(bias.astype(np.float32)))

        for p in self.parameters(): p.requires_grad_(False)

        print '[PCA] explained variance: {}'.format(sum(main.explained_variance_ratio_))

    def forward(self, x):
        # return self.main(x)
        return _forward(self.main, x)