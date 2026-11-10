"""Apply the three minimal host hooks, with hashes, backups and optional rollback."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path

API_HOOK = '\n# Optional, isolated AIS screening workbench.\nfrom anomaly_workbench import attach as attach_anomaly_workbench\nattach_anomaly_workbench(app)\n'
NAV_HOOK = '''      <a class="new-chat workspace-entry" href="/anomaly-workbench" id="anomaly-workbench-entry">
        <span class="workspace-entry-icon" aria-hidden="true">⌖</span>
        <span class="workspace-entry-copy"><strong>异常行为初筛与核查台</strong><small>物理流场解耦 · 候选排序 · 证据回溯</small></span>
        <span class="workspace-entry-arrow" aria-hidden="true">→</span>
      </a>
'''
MAP_HOOK = '      <a href="/anomaly-workbench" class="btn-briefing" id="anomaly-workbench-map-entry" style="text-decoration:none" title="打开独立的异常行为初筛与物理证据核查台">⚡ 异常初筛核查</a>\n'


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def changes(src: Path) -> dict[Path,str]:
    app=src/'maritime_agent';result={}
    api=app/'api.py';text=api.read_text(encoding='utf-8')
    if 'attach_anomaly_workbench(app)' not in text:
        tree=ast.parse(text)
        assign=next((n for n in tree.body if isinstance(n,ast.Assign) and
                     any(isinstance(t,ast.Name) and t.id=='app' for t in n.targets) and
                     isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='FastAPI'),None)
        if assign is None:raise ValueError('找不到 FastAPI app 创建位置，未修改文件')
        lines=text.splitlines(keepends=True);lines.insert(assign.end_lineno,API_HOOK)
        result[api]=''.join(lines)
    nav=app/'ui.html';text=nav.read_text(encoding='utf-8')
    if 'id="anomaly-workbench-entry"' not in text:
        lines=text.splitlines(keepends=True)
        index=next((i for i,l in enumerate(lines) if 'id="open-critiques-btn"' in l),None)
        if index is None:raise ValueError('找不到自进化知识库入口，未修改文件')
        lines.insert(index+1,NAV_HOOK);result[nav]=''.join(lines)
    page=app/'wind_map.html';text=page.read_text(encoding='utf-8')
    if 'id="anomaly-workbench-map-entry"' not in text:
        lines=text.splitlines(keepends=True)
        index=next((i for i,l in enumerate(lines) if 'id="btn-open-briefing"' in l),None)
        if index is None:raise ValueError('找不到态势研判简报入口，未修改文件')
        lines.insert(index,MAP_HOOK);result[page]=''.join(lines)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app-src',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--rollback',type=Path,help='此前生成的 integration.json；当前文件须仍与挂载后哈希一致')
    args=parser.parse_args();src=args.app_src.resolve()
    if args.rollback:
        manifest=json.loads(args.rollback.read_text(encoding='utf-8'))
        entries=manifest['files']
        for e in entries:
            p=src/e['path']
            if digest(p.read_bytes())!=e['after_sha256']:
                raise ValueError(f'{p.name} 在挂载后又有更新，停止自动回退')
        if not args.dry_run:
            for e in entries:(src/e['path']).write_bytes((args.rollback.parent/e['backup']).read_bytes())
        print('回退检查通过' if args.dry_run else '已回退三个挂载文件；模块与复核记录保留')
        return
    if not (src/'anomaly_workbench/__init__.py').is_file():
        raise ValueError('请先把 anomaly_workbench 文件夹完整复制到应用 src 目录')
    plan=changes(src)
    if not plan:print('入口与路由已经挂载，无需重复修改');return
    for p,content in plan.items():
        print(''.join(difflib.unified_diff(p.read_text(encoding='utf-8').splitlines(True),content.splitlines(True),
              fromfile=str(p.relative_to(src)),tofile=str(p.relative_to(src)))),end='')
    if args.dry_run:return
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder=src/'anomaly_workbench/runtime/integration-backups'/stamp;folder.mkdir(parents=True)
    records=[]
    for p,content in plan.items():
        before=p.read_bytes();after=content.encode('utf-8');backup=p.name+'.original'
        (folder/backup).write_bytes(before)
        records.append({'path':str(p.relative_to(src)),'backup':backup,
                        'before_sha256':digest(before),'after_sha256':digest(after)})
    # All anchors were validated before the first mutation.
    for p,content in plan.items():p.write_bytes(content.encode('utf-8'))
    manifest=folder/'integration.json';manifest.write_text(json.dumps({'files':records},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'挂载完成；回退记录：{manifest}')


if __name__=='__main__':main()
