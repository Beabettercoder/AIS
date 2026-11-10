import pathlib,hashlib,json,sys
root=pathlib.Path(sys.argv[1]); files={}
for f in sorted(root.rglob('*')):
 if not f.is_file() or '__pycache__' in f.parts or f.name.startswith('manifest_'):continue
 h=hashlib.sha256()
 with f.open('rb') as a:
  for b in iter(lambda:a.read(1048576),b''):h.update(b)
 files[str(f.relative_to(root)).replace('\\','/')]=h.hexdigest()
print(json.dumps(files,sort_keys=True,indent=2))
