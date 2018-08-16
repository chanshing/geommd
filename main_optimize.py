import os
import argparse
import time
import scipy.signal as signal
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import models
import utils
import logger
import mmd


def main(args):
    utils.seedme(args.seed)
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() and not args.nocuda else 'cpu')

    os.system('mkdir -p {}'.format(args.outf))

    img = utils.load_image(args.image, resize=args.resize) # (channel, height, width), [-1,1]
    x0 = torch.from_numpy(img).unsqueeze(0).to(device)  # (1, channel, height, width), torch

    args.img = img
    args.nc = img.shape[0]

    x = models.X(image_size=args.syn_size, nc=args.nc, batch_size=args.batch_size).to(device)
    optimizer = optim.Adam(x.parameters(), lr=args.lr)

    netE = models.choose_archE(args).to(device)
    print netE

    mmdrq = mmd.MMDrq(nu=args.nu, encoder=netE)
    loss_func = utils.Loss(x0, mmdrq, args.patch_size, args.npatch)

    losses = []
    start_time = time.time()
    for i in range(args.niter):
        optimizer.zero_grad()

        x1 = x()
        loss = loss_func(x1).mean()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (i+1) % 500 == 0:
            print '[{}/{}] loss: {}'.format(i+1, args.niter, loss.item())
            fig, ax = plt.subplots()
            ax.plot(signal.medfilt(losses, 101)[50:-50])
            ax.set_yscale('symlog')
            fig.tight_layout()
            fig.savefig('{}/loss.png'.format(args.outf))
            plt.close(fig)
            logger.vutils.save_image(x1, '{}/x_{}.png'.format(args.outf, i+1), normalize=True, nrow=10)
            print 'This round took {0} secs'.format(time.time()-start_time)
            start_time = time.time()

    np.save('{}/x1.npy'.format(args.outf), x1.detach().cpu().numpy().squeeze())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='img/peppers.jpg')
    parser.add_argument('--resize', type=int, default=None)
    parser.add_argument('--syn_size', type=int, default=256)
    parser.add_argument('--outf', default='tmp/optimize/peppers')
    parser.add_argument('--nocuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--niter', type=int, default=5000)
    parser.add_argument('--archE', default='pca')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--npatch', type=int, default=128)
    parser.add_argument('--ncode', type=int, default=16)
    parser.add_argument('--ndf', type=int, default=16)
    parser.add_argument('--netE', default=None)
    parser.add_argument('--nu', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=1)

    main(parser.parse_args())
