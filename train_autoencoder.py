import os
import argparse
import scipy.signal

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import models
import utils
import logger


def main(args):
    utils.seedme(args.seed)
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() and not args.nocuda else 'cpu')
    
    os.system('mkdir -p {}'.format(args.outf))

    dataloader_train = utils.get_patchloader(args.image_train, resize=args.resize_train, patch_size=args.patch_size, batch_size=args.batch_size_train,
                                             fliplr=args.fliplr, flipud=args.flipud, rot90=args.rot90, smooth=args.smooth)
    if args.image_valid:
        dataloader_valid = utils.get_patchloader(args.image_valid, resize=args.resize_valid, patch_size=args.patch_size, batch_size=args.batch_size_valid,
                                                fliplr=args.fliplr, flipud=args.flipud, rot90=args.rot90, smooth=args.smooth)

    netG = models.DCGAN_G(image_size=args.patch_size, nc=args.nc, nz=args.ncode, ngf=args.ngf).to(device)
    netE = models.Encoder(patch_size=args.patch_size, nc=args.nc, ncode=args.ncode, ndf=args.ndf).to(device)

    print netG
    print netE

    optimizer = optim.Adam(list(netG.parameters()) + list(netE.parameters()), lr=args.lr, amsgrad=True)
    loss_func = nn.MSELoss()

    losses = []
    losses_valid = []
    best_loss = 1e16
    for i in range(args.niter):
        optimizer.zero_grad()
        x = next(dataloader_train).to(device)
        if args.sigma:
            x = utils.add_noise(x, args.sigma)
        y = netG(netE(x))
        loss = loss_func(y, x)
        loss.backward()
        optimizer.step()

        if args.image_valid:
            with torch.no_grad():
                netG.eval()
                netE.eval()
                x_ = next(dataloader_valid).to(device)
                if args.sigma:
                    x_ = utils.add_noise(x, args.sigma)
                y_ = netG(netE(x_))
                loss_valid = loss_func(y_, x_)
                netG.train()
                netE.train()
                losses_valid.append(loss_valid.item())

        _loss = loss_valid.item() if args.image_valid else loss.item()
        if _loss + 1e-3 < best_loss:
            best_loss = _loss
            print "[{}/{}] best loss: {}".format(i+1, args.niter, best_loss)
            if args.save_best:
                torch.save(netE.state_dict(), '{}/netD_best.pth'.format(args.outf))

        losses.append(loss.item())
        if (i+1) % args.nprint == 0:
            if args.image_valid:
                print '[{}/{}] train: {}, test: {}, best: {}'.format(i+1, args.niter, loss.item(), loss_valid.item(), best_loss)
            else:
                print '[{}/{}] train: {}, best: {}'.format(i+1, args.niter, loss.item(), best_loss)
            logger.vutils.save_image(torch.cat([x,y], dim=0), '{}/train_{}.png'.format(args.outf, i+1), normalize=True)
            fig, ax = plt.subplots()
            ax.semilogy(scipy.signal.medfilt(losses, 11)[5:-5], label='train')
            if args.image_valid:
                logger.vutils.save_image(torch.cat([x_,y_], dim=0), '{}/test_{}.png'.format(args.outf, i+1), normalize=True, nrow=32)
                ax.semilogy(scipy.signal.medfilt(losses_valid, 11)[5:-5], label='valid')
            fig.legend()
            fig.savefig('{}/loss.png'.format(args.outf))
            plt.close(fig)
            torch.save(netE.state_dict(), '{}/netD_iter_{}.pth'.format(args.outf, i+1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument('--outf', default='tmp/peppers')
    parser.add_argument('--nocuda', action='store_true')
    parser.add_argument('--niter', type=int, default=2000)
    parser.add_argument('--nprint', type=int, default=100)
    parser.add_argument('--image_train', default='img/peppers.jpg')
    parser.add_argument('--image_valid', default=None)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--resize_train', type=int, default=None)
    parser.add_argument('--resize_valid', type=int, default=None)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--batch_size_train', type=int, default=32)
    parser.add_argument('--batch_size_valid', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ncode', type=int, default=16)
    # decoder
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--g_extra', type=int, default=0)
    # aurgmentation
    parser.add_argument('--fliplr', type=bool, default=False)
    parser.add_argument('--flipud', type=bool, default=False)
    parser.add_argument('--rot90', type=bool, default=False)
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--smooth', type=float, default=None)
    # encoder
    parser.add_argument('--ndf', type=int, default=32)

    main(parser.parse_args())
