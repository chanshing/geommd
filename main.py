import argparse
import time

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import models
import utils
import logger
import mmd


def main(args):
    utils.seedme(args.seed)
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() and not args.nocuda else 'cpu')

    img = utils.load_image(args.image, resize=args.resize) # (channel, height, width), [-1,1]
    x0 = torch.from_numpy(img).unsqueeze(0).to(device)  # (1, channel, height, width), torch

    args.img = img
    args.nc = img.shape[0]

    netG = models.choose_archG(args).to(device)
    netE = models.choose_archE(args).to(device)
    print netE
    print netG

    optimizer = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), amsgrad=True)
    z = torch.randn(args.batch_size,args.nz,1,1).to(device)

    mmdrq = mmd.MMDrq(nu=args.nu, encoder=netE)
    loss_func = utils.Loss(x0, mmdrq, args.patch_size, args.npatch)

    log = logger.Logger(args, netG, netE)
    log.save_image(x0, 'ref.png')
    nstart, nend = log.nstart, log.nend

    start_time = time.time()
    for i in range(nstart, nend):
        optimizer.zero_grad()

        x1 = netG(z.normal_())
        loss = loss_func(x1).mean()
        ent = utils.sample_entropy(x1.view(x1.shape[0],-1))
        kl = loss - args.alpha*ent

        kl.backward()
        optimizer.step()

        # --- logging
        log.log(loss.item(), ent.item(), kl.item())
        if (i+1) % 500 == 0:
            print 'This round took {0} secs'.format(time.time()-start_time)
            start_time = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nocuda', action='store_true')
    parser.add_argument('--outf', default='tmp/peppers')
    parser.add_argument('--image', default='img/peppers.jpg')
    parser.add_argument('--resize', type=int, default=None)
    parser.add_argument('--niter', type=int, default=5000)
    parser.add_argument('--archE', default='autoencoder')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--syn_size', type=int, default=256)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--g_extra', type=int, default=0)
    parser.add_argument('--nz', type=int, default=256)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--npatch', type=int, default=128)
    parser.add_argument('--ncode', type=int, default=8)
    parser.add_argument('--ndf', type=int, default=32)
    parser.add_argument('--netE', default=None)
    parser.add_argument('--nu', type=float, default=0.5)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--archG', default='v0')

    main(parser.parse_args())
