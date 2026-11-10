from pathlib import Path
import hashlib, json

base=Path('E:/AIS/outputs')
out=base/'relation_latest_report_20261030'
builder=(base/'relation_repair_20261030/builder.py').read_text(encoding='utf-8')
def replace_once(text, old, new):
    assert text.count(old)==1, (old,text.count(old))
    return text.replace(old,new)
builder=replace_once(builder,"OUT=BASE/'relation_repair_20261030'","OUT=BASE/'relation_latest_report_20261030'")
builder=replace_once(builder,"selected=np.flatnonzero((nt>=phase_start)&(nt<=t[j]))\n            if len(selected)==0: selected=np.array([q])", "selected=np.array([q])  # fixed latest aligned report; both F0 and F1")
builder=replace_once(builder,"'scope':'development, no model training; paired zero-order fallback and causal segment report aggregation'", "'scope':'development, latest aligned report for both arms, paired zero-order fallback'")
builder=replace_once(builder,"'rho':'shared min(1, n/4) multiplied by mean absolute reliability of aligned members; zero when fit invalid'", "'rho':'F1 min(1,n/4)*mean w of latest aligned reports; F0 model uses min(1,n/4); positive on zero-order fallback; zero only when no estimate'")
(out/'builder.py').write_text(builder,encoding='utf-8')
src=(base/'tcn_suffix_stability_20261030/train_multiseed.py').read_text(encoding='utf-8')
src=replace_once(src,"OUT=Path('/mnt/sda/WangZijia/AIS/outputs/tcn_suffix_stability_20261030')", "OUT=Path('/mnt/sda/WangZijia/AIS/outputs/relation_latest_report_20261030')")
src=replace_once(src,"OLD=Path('/mnt/sda/WangZijia/AIS/outputs/relation_repair_20261030')", "OLD=Path('/mnt/sda/WangZijia/AIS/outputs/relation_latest_report_20261030')")
src=replace_once(src,"'epochs':25", "'epochs':(75 if name=='itransformer' else 25)")
src=replace_once(src,"'training':'BCE on base only; fixed epoch25'", "'training':'BCE on base only; fixed epoch75 itransformer, epoch25 tcn'")
src=replace_once(src,"'F0':'causal target plus distance-weighted segment mean field'", "'F0':'causal target plus distance-weighted latest-report field'")
src=replace_once(src,"'F1':'causal target plus reliability-weighted segment report mean and distance*member-reliability weighted field'", "'F1':'causal target plus latest-report field weighted distance*report reliability'")
src=replace_once(src,"for epoch in range(25):", "for epoch in range(75 if name=='itransformer' else 25):")
src=replace_once(src,"'last_epoch':25", "'last_epoch':(75 if name=='itransformer' else 25)")
(out/'train.py').write_text(src,encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol={'frozen_before_candidate_train_or_candidate_validation_scores':True,'mechanism':'latest aligned member report q <= target time and age <=300 s, identical selection in F0 and F1; F0 distance-weighted, F1 distance times w_abs, latest-report w_abs; retain common affine/zero-order fallback','why':'previous trace shows suffix dilution and w_abs detects part of speed changes; no claim of guaranteed outcome','comparisons':'original cumulative versus latest, plus F1 versus F0 within each; no selection by validation scores','seeds':[0,1,2],'backbones':{'tcn':25,'itransformer':75},'arms':['C0','F0','F1'],'views':['base','natural','suffix','combined_l100_m600'],'train':'base only; fixed final epoch; same initialization and minibatches across arms; lr .001 AdamW, batch 128, loss BCE','normalization':'train base only with pooled F0/F1 context, all arms shared; same 35-d interface','metric':'pooled AP with target-MMSI paired 2000 bootstrap; report all seed results; CI over fixed models only','original':'relation_repair_20261030 plus heading_alignment_20261030/tcn_suffix_stability_20261030/itrans_stability_20261030','scope':'exposed development split only; no test set labels','sha256':{'builder':sha(out/'builder.py'),'train':sha(out/'train.py')}}
(out/'frozen_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(protocol,ensure_ascii=False,indent=2))


