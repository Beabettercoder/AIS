import train, numpy as np, torch
d=train.read('val'); mu,sd=train.fit(d)
for arm in train.ARMS:
    x,m,t=train.prep(d,'base',arm,mu,sd,'cpu')
    assert x.shape==(1088,20,24) and np.isfinite(x.numpy()).all()
    assert np.array_equal(x.numpy()[:, :, :8],train.prep(d,'natural',arm,mu,sd,'cpu')[0].numpy()[:,:,:8])
    if arm=='C0': assert np.all(x.numpy()[:,:,8:]==0)
    for kind in ('tcn','gru','transformer','itransformer'):
        model=train.make_model(kind,24,max_len=20,hidden=64)
        out=model(x[:8],m[:8],t[:8])['logits'];out.sum().backward();assert torch.isfinite(out).all()
    print(arm,'shape',x.shape,'finite',bool(torch.isfinite(x).all()))
print('post-diff/rate standardization',mu.tolist(),sd.tolist())
