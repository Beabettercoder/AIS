import json
p='/mnt/sda/WangZijia/AIS/outputs/heading_alignment_20261030/development_summary.json';a=json.load(open(p));
for n,r in a['models'].items():
 print(n)
 for v in ('base','natural','suffix','combined_l100_m600'):
  d=r['values'][v]; i=r['intervals'][v]['f1_vs_f0'];q=r['intervals'][v]['f1_vs_c0']; print(v,'F0',round(d['F0']['pooled_AP'],4),'F1',round(d['F1']['pooled_AP'],4),'diffCI',round(100*i['lo'],2),round(100*i['hi'],2),'vsC0CI',round(100*q['lo'],2),round(100*q['hi'],2))
