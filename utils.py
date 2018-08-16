import random
import numpy as np
import scipy.stats as stats

from PIL import Image

import torch

def seedme(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def preprocess_image(img):
    """ to numpy, channel first, [-1,1] """
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2: img = np.expand_dims(img, axis=-1)  # img is grayscale
    img = np.transpose(img, (2, 0, 1))
    img = 2.0*(img/255.0 - 0.5)  # [-1, 1]
    return img

def load_image(fname, resize):
    """ load, resize, preprocess """
    img = Image.open(fname)
    if resize:
        img = img.resize((resize, resize))
    img = preprocess_image(img)  # (channel, height, width), [-1,1]
    return img

def get_patches(img, patch_size):
    """ Return all patches from img """
    nc, ny, nx = img.shape
    my, mx = ny - patch_size + 1, nx - patch_size + 1

    patches = np.empty((my*mx, nc, patch_size, patch_size))

    for i in range(my):
        for j in range(mx):
            patches[i*mx + j] = img[:, i:i+patch_size,j:j+patch_size]

    np.random.shuffle(patches)

    return patches

def choice_patches(img, patch_size, size=128):
    """ Random pick patches from img """
    _, ny, nx = img.shape
    my, mx = ny - patch_size + 1, nx - patch_size + 1

    patches = []
    for _ in range(size):
        i = np.random.choice(range(my))
        j = np.random.choice(range(mx))
        patches.append(img[:, i:i+patch_size, j:j+patch_size])

    return torch.stack(patches)

def batch_choice_patches(imgs, patch_size, size=128):
    """ (batch) Random pick patches from img """
    b, nc, ny, nx = imgs.shape
    my, mx = ny - patch_size + 1, nx - patch_size + 1

    patches = []
    for img in imgs:
        ii = np.random.choice(range(my), size=size)
        jj = np.random.choice(range(mx), size=size)
        patches.extend([img[:, i:i+patch_size, j:j+patch_size] for i,j in zip(ii,jj)])

    return torch.stack(patches).view(b, size, nc, patch_size, patch_size)

def get_patchloader(fname, patch_size, resize=None, batch_size=32, 
                      fliplr=False, flipud=False, rot90=False, smooth=None):
    image = Image.open(fname)
    if resize:
        image = image.resize((resize,resize))
    img = preprocess_image(image)
    _, ny, nx = img.shape
    my, mx = ny - patch_size + 1, nx - patch_size + 1
    count = 0

    while True:
        count += 1

        batch = []
        for _ in range(batch_size):
            i = np.random.choice(range(my))
            j = np.random.choice(range(mx))
            patch = img[:, i:i+patch_size, j:j+patch_size]
            patch = random_transform(patch, fliplr, flipud, rot90)
            batch.append(patch)
        batch = np.stack(batch)  # (batch_size, channel, height, width)
        batch = torch.from_numpy(batch)

        if smooth:
            batch = gauss_smooth_binary(batch, smooth)

        yield batch

def random_transform(img, fliplr, flipud, rot90):
    if fliplr and np.random.choice((True, False)):
        img = np.flip(img, axis=2)
    if flipud and np.random.choice((True, False)):
        img = np.flip(img, axis=1)
    if rot90:
        img = np.rot90(img, k=np.random.choice((0,1,-1)), axes=(1,2))  # -1 means clockwise
    return img

def add_noise(data, sigma):
    noise = torch.randn_like(data)*sigma
    data = (data + noise).clamp_(-1,1)
    return data

def gauss_smooth_binary(data, sigma):
    """ Smooth binary data with Gaussian noise. Binary values -1 or 1 """
    noise = stats.halfnorm.rvs(size=data.shape, scale=sigma).astype(np.float32)
    noise = torch.from_numpy(noise)
    data = (data - data*noise).clamp_(-1,1)
    return data

class Loss(object):
    def __init__(self, x0, mmd_func, patch_size, npatch, padding=True):
        self.x0 = x0
        self.mmd_func = mmd_func
        self.patch_size = patch_size
        self.npatch = npatch
        self.padding = padding
        if padding:
            self.pad = torch.nn.ReflectionPad2d(patch_size/2)

    def __call__(self, x):
        if self.padding:
            x = self.pad(x)
        x0 = self.x0.expand(x.shape[0], *self.x0.shape[1:])
        patches0 = batch_choice_patches(x0, patch_size=self.patch_size, size=self.npatch)
        patches1 = batch_choice_patches(x, patch_size=self.patch_size, size=self.npatch)
        return self.mmd_func(patches0, patches1)

def sample_entropy(sample):
    """ Estimator based on kth nearest neighbor, A new class of random vector entropy estimators (Goria et al.) """
    sample = sample.view(sample.size(0), -1)
    m, n = sample.shape

    mat_ = distance_matrix(sample)
    mat, _ = mat_.sort(dim=1)
    k = int(np.round(np.sqrt(sample.size(0))))  # heuristic
    rho = mat[:,k]  # kth nearest
    entropy = 0.5*(rho + 1e-16).log().sum()
    entropy *= float(n)/m

    return entropy

def distance_matrix(sample1, sample2=None):
    """ Build (squared) distance matrix.
    Shapes of sample1 and sample2 must be (m x k) and (n x k), respectively. (feature last)
    If sample2 is None, sample2 = sample1. """

    if sample2 is None:
        m = sample1.shape[0]
        sample_norm2 = sample1.mul(sample1).sum(dim=1).view(-1,1)
        sample_norm2 = sample_norm2.expand(m, m)
        mat = sample1.mm(sample1.t()).mul(-2) + sample_norm2.add(sample_norm2.t())
    else:
        m, n = sample1.shape[0], sample2.shape[0]
        sample1_norm2 = sample1.mul(sample1).sum(dim=1).view(-1,1)
        sample1_norm2 = sample1_norm2.expand(m, n)
        sample2_norm2 = sample2.mul(sample2).sum(dim=1).view(-1,1)
        sample2_norm2 = sample2_norm2.expand(n, m)
        mat = sample1.mm(sample2.t()).mul(-2) + sample1_norm2.add(sample2_norm2.t())

    return mat.clamp(0)
