"""Validate frozen latest-report feature and selected split digests, never loads arrays."""
from pathlib import Path
import json,hashlib
CAND=Path('/mnt/sda/WangZijia/AIS/outputs/relation_latest_report_20261030')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def check_candidate(splits=('train','val')):
 f=json.loads((CAND/'frozen_protocol.json').read_text(encoding='utf8'))
 for name in ('builder','train'):
  if sha(CAND/(name+'.py'))!=f['sha256'][name]: raise ValueError('Frozen candidate code changed: '+name)
 for split in splits:
  meta=json.loads((CAND/(split+'_relation_field.json')).read_text(encoding='utf8'))
  if meta['builder_sha256']!=f['sha256']['builder']: raise ValueError('Candidate builder mismatch: '+split)
  if sha(CAND/(split+'_relation_field.npz'))!=meta['file_sha256']: raise ValueError('Candidate data mismatch: '+split)
 return f

def check_gate_candidate(approval,freeze):
 if approval.get('relation_candidate_name')!='latest_report_20261030': raise ValueError('Candidate not registered in gate')
 for name in ('builder','train'):
  if approval.get('relation_candidate_'+name+'_sha256')!=freeze['sha256'][name]: raise ValueError('Candidate gate digest mismatch: '+name)

 for split in ('train','val'):
  m=json.loads((CAND/(split+'_relation_field.json')).read_text(encoding='utf8'))
  if approval.get('relation_candidate_'+split+'_data_sha256')!=m['file_sha256']: raise ValueError('Candidate gate data digest mismatch: '+split)
