import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn

class Adapter(nn.Module):
    def __init__(self, seq_len, channels, gpu=0, **kwargs):
        super().__init__()
        if seq_len % 2: raise ValueError('TFT-GCN original inverse FFT requires an even sequence length')
        root = Path(__file__).resolve().parents[1] / 'vendor' / 'TFT-GCN'
        sys.path.insert(0, str(root))
        from models.model import Model
        c = SimpleNamespace(seq_len=seq_len,enc_in=channels,c_out=channels,top_k=3,d_model=64,d_ff=128,
            e_layers=2,dropout=.1,n_heads=4,gpu=gpu,hidden=8,gdepth=2,alpha=.3,k=5,nvechidden=8,
            fnet_d_model=64,fnet_d_ff=128,fnet_layers=2,complex_dropout=.1)
        self.model = Model(c)
    def loss(self, x, indices=None, epoch=0):
        out=self.model(x)
        ft1=torch.fft.fft(out.transpose(1,2),norm='forward')
        ft2=torch.fft.fft(x.transpose(1,2),norm='forward')
        return ((out-x)**2).mean()+((ft1.real-ft2.real)**2).mean()+((ft1.imag-ft2.imag)**2).mean()
    def score(self, x):
        return ((self.model(x)-x)**2).mean(dim=(1,2))
