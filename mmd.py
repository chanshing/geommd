class MMDrq(object):
    def __init__(self, nu=0.5, encoder=None, biased=False):
        self.nu = nu
        self.encoder = encoder
        self.biased = biased

    def __call__(self, x, y):
        if self.encoder:
            x, y = self.encoder(x), self.encoder(y)
        
        # (batch_size, npatch, ncode)
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        y = y.view(y.shape[0], y.shape[1], y.shape[2])

        dyy = batch_distance_matrix(y, y)
        dxy = batch_distance_matrix(x, y)

        l = (dxy.clamp(0).sqrt().median().item()) + 1e-16  # median heuristic
        kyy = dyy.div(2*self.nu*l**2).add(1).pow(-self.nu)
        kxy = dxy.div(2*self.nu*l**2).add(1).pow(-self.nu)

        kxy = kxy.view(kxy.shape[0], -1).mean(-1)

        if self.biased:
            kyy = kyy.view(kyy.shape[0], -1).mean(-1)
        else:
            _, m, m = kyy.shape
            diagsum = kyy[:,range(m),range(m)].sum(dim=-1)
            kyy = kyy.view(kyy.shape[0], -1).sum(-1)
            kyy = (kyy - diagsum)/(m*(m-1))

        return kyy - 2*kxy

def batch_distance_matrix(sample1, sample2=None):
    """ Compute (squared) distance matrix. 
    Shapes of sample1 and sample2 must be (b x m x k) and (b x n x k), respectively. (feature last)
    If sample2 is None, sample2 = sample1. """

    if sample2 is None:
        b, m, _ = sample1.shape
        sample_norm2 = (sample1*sample1).sum(dim=2, keepdim=True)
        sample_norm2 = sample_norm2.expand(b, m, m)
        # broadcast
        mat = -2*sample1.matmul(sample1.transpose(1,2)) + (sample_norm2 + sample_norm2.transpose(1,2))
    else:
        b, m, _ = sample1.shape
        b, n, _ = sample2.shape
        sample1_norm2 = (sample1*sample1).sum(dim=2, keepdim=True)
        sample1_norm2 = sample1_norm2.expand(b, m, n)
        sample2_norm2 = (sample2*sample2).sum(dim=2, keepdim=True)
        sample2_norm2 = sample2_norm2.expand(b, n, m)
        # broadcast
        mat = -2*sample1.matmul(sample2.transpose(1,2)) + (sample2_norm2.transpose(1,2) + sample1_norm2)

    return mat.clamp(0)

