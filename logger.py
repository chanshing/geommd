import os
import re
import numpy as np
import scipy.signal as signal

import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

import torch
import torchvision.utils as vutils

class Logger(object):
    def __init__(self, args, netG, netD):
        outf = '{}/{}_ngf{}_x{}/alpha_{}/b{}_lr{}_beta{}'.format(args.outf, args.archG, args.ngf, args.g_extra, args.alpha, args.batch_size, args.lr, args.beta1)
        os.system('mkdir -p {}'.format(outf))

        # FORMAT = '%(asctime)s %(levelname)s: %(message)s'
        # logging.basicConfig(filename='{}/log.log'.format(outf), level=logging.INFO, format=FORMAT, datefmt='%I:%M:%S %p')

        if args.resume:
            niter = get_last_iter(outf)
            assert (niter > 0), 'last model not found'
            # logging.warn('Found last model, loading: netG, netD, zfix')
            netG.load_state_dict(torch.load('{}/netG_iter_{}.pth'.format(outf, niter)))
            netD.load_state_dict(torch.load('{}/netD.pth'.format(outf)))
            zfix = torch.load('{}/zfix.pth'.format(outf))

            nstart = niter
            nend = niter + args.niter
            loss = list(np.load('{}/loss.npy'.format(outf)))
            ent = list(np.load('{}/ent.npy'.format(outf)))
            kl = list(np.load('{}/kl.npy'.format(outf)))

        else:
            loss, ent, kl = [], [], []
            nstart, nend = 0, args.niter
            zfix = torch.randn(16, args.nz, 1, 1).to(next(netG.parameters()).device)  # for plot
            torch.save(netD.state_dict(), '{}/netD.pth'.format(outf))
            torch.save(zfix, '{}/zfix.pth'.format(outf))

        self.outf = outf
        self.netG, self.netD = netG, netD
        self.loss, self.ent, self.kl = loss, ent, kl
        self.nstart, self.nend = nstart, nend
        self.zfix = zfix
        self.iter = nstart

    def log(self, loss, ent, kl):
        self.iter += 1

        self.loss.append(loss)
        self.ent.append(ent)
        self.kl.append(kl)

        if (self.iter) % 100 == 0:
            print '[{}/{}] loss: {}, ent: {}, kl: {}'.format(self.iter, self.nend, loss, ent, kl)

        if (self.iter) % 500 == 0:
            fig, axs = plt.subplots(3)
            axs[0].plot(signal.medfilt(np.array(self.loss), 101)[1000:-50])
            axs[1].plot(signal.medfilt(np.array(self.ent), 101)[1000:-50])
            axs[2].plot(signal.medfilt(np.array(self.kl), 101)[1000:-50])
            for ax in axs: ax.set_yscale('symlog')
            for ax, ylabel in zip(axs, ['loss', 'entropy', 'KL']): ax.set_ylabel(ylabel)
            fig.tight_layout()
            fig.savefig('{}/loss.png'.format(self.outf))
            plt.close(fig)

            self.netG.eval()
            with torch.no_grad():
                vutils.save_image(self.netG(self.zfix), '{}/synthesis_{}.png'.format(self.outf, self.iter), normalize=True, nrow=4, padding=10)
            self.netG.train()
            torch.save(self.netG.state_dict(), '{}/netG_iter_{}.pth'.format(self.outf, self.iter))
            np.save('{}/loss.npy'.format(self.outf), np.asarray(self.loss))
            np.save('{}/ent.npy'.format(self.outf), np.asarray(self.ent))
            np.save('{}/kl.npy'.format(self.outf), np.asarray(self.kl))

    def save_image(self, x, fname):
        vutils.save_image(x, '{}/{}'.format(self.outf, fname), normalize=True)

def extract_iter(f):
    s = re.findall(r"netG_iter_(\d+).pth", f)
    return int(s[0]) if s else -1

def get_last_iter(outf):
    files = os.listdir(outf)
    return max(extract_iter(f) for f in files)

# def plot_sample(x, ij0, ij1, outf, i):
#     nrows=2
#     ncols=4
#     fig = plt.figure(figsize=(10, 10*nrows/ncols))
#     fig.subplots_adjust(top=1,right=1,bottom=0,left=0, hspace=.1, wspace=.01)
#     axs = fig.subplots(nrows, ncols).ravel()
#     for x_, ax in zip(x, axs):
#         ax.xaxis.set_visible(False)
#         ax.yaxis.set_visible(False)
#         ax.xaxis.set_major_locator(NullLocator())
#         ax.yaxis.set_major_locator(NullLocator())
#         ax.imshow(x_, origin='lower')
#         ax.scatter(ij1[0], ij1[1], marker='o', s=20)
#         ax.scatter(ij0[0], ij0[1], marker='x', s=20)
#         ax.set_xlim(0,127)
#         ax.set_ylim(0,127)
#     fig.savefig('{}/sample_{}.png'.format(outf, i), bbox_inches='tight', pad_inches=0)
#     plt.close(fig)
