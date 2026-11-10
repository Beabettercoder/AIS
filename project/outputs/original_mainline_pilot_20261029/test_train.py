from pilot_train import *
torch.set_num_threads(1);d=load('train');n=fit(d);t=convert(d,n,torch.device('cpu'));ix=slice(0,8)
for kind in ['tcn','gru','transformer','itransformer']:
 seed(0);m=Model(kind);x=t['self'][ix];mask=t['mask'][ix];tm=t['times'][ix];fi=t['f1'][ix];rho=t['rho'][ix]
 m.eval();p1=m(x,mask,tm,fi,rho,False);p2=m(x,mask,tm,fi,rho,True);assert torch.equal(p1[0],p2[0])
 p3=m(x,mask,tm,fi,torch.zeros_like(rho),True);assert torch.equal(p1[0],p3[0]);loss,log=historical_bundle(m.backbone.head,m.typehead,m.ctrl,p2[0],p2[1],p2[2],p2[3],t['state1'][ix],mask,t['y'][ix],t['yc'][ix],torch.ones(5),torch.ones(8));loss.backward();assert torch.isfinite(loss);print(kind,'loss',float(loss),'epsilon',float(log['epsilon'].mean()))
