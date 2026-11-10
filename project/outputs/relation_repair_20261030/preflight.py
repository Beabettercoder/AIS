import train,numpy as np,json
a=train.read('train');b=train.read('val')
assert not (set(a['mmsi'])&set(b['mmsi']))
assert not (set(a['source'])&set(b['source']))
mu,sd=train.fit(a)
assert np.isfinite(mu).all() and np.isfinite(sd).all() and (sd>0).all()
x0,cm=train.features(b,'base','C0')
assert (x0[:,:,8:]==0).all() and not cm.any()
for key in train.ARMS:
 x,mask,t=train.prep(b,'base',key,mu,sd,'cpu')
 assert np.isfinite(x).all()
print(json.dumps({'train_windows':len(a['y']),'val_windows':len(b['y']),'mmsi_intersect':0,'source_intersect':0,'post_diff_training_rate_sd':sd[[12,13,18,19]].tolist(),'test_read':False}))
