import torch
from carla import Adapter
torch.set_num_threads(1)
m=Adapter(20,13)
x=torch.randn(4,20,13)
loss=m.loss(x,positive=x+0.01*torch.randn_like(x)); loss.backward(); print('pretext',float(loss))
m.zero_grad();m.begin_clustering()
loss=m.loss(x,near=x,far=x.flip(0));loss.backward();print('clustering',float(loss))
m.select_normal_cluster(x);print('score',m.score(x))
