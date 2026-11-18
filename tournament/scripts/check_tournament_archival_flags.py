from pathlib import Path
import json,subprocess,tempfile,sys,shutil
root=Path.cwd();base=root/'benchmarks/tournament-2026-11-04';media=root/'local-evidence/tournament-2026-11-04/video';data=json.load(open(root/'tournament-site/dist/data.json'))
allvideos={}
for v in data['videos']+json.load(open(media/'swiss-4-public-catalog-archived.json'))+json.load(open(media/'lower-before-wave4-public-catalog-archived.json')):
 if v['stage'] in ('swiss-4','playoff-1','playoff-2','playoff-3'):allvideos[(v['stage'],v['matchA'],v['matchB'],v['game'])]=v
assert len(allvideos)==34
results=[]
with tempfile.TemporaryDirectory(prefix='tournament-archival-check-',dir='/private/tmp') as temp:
 p=Path(temp);out=p/'output';view=p/'media';site=p/'site';out.mkdir();view.mkdir();site.mkdir()
 for f in media.iterdir():
  if f.is_file() and f.suffix in ('.mp4','.json','.webp','.vtt'): (view/f.name).symlink_to(f)
 # The view's proof must be a fresh writable file, not a link to private evidence.
 for name in ('replay-catalog-verification.json','swiss-4-public-catalog-archived.json','lower-before-wave4-public-catalog-archived.json'):
  (view/name).unlink(missing_ok=True)
 for stage in ('swiss-4','playoff-1','playoff-2','playoff-3'):
  shutil.copy2(base/(stage+'.json'),out/(stage+'.json'));(out/stage).symlink_to(base/stage,target_is_directory=True)
 data['videos']=list(allvideos.values());(site/'data.json').write_text(json.dumps(data))
 command=[sys.executable,str(root/'tournament/scripts/render_tournament_replays.py'),'--output',str(out),'--media-dir',str(view),'--site',str(site)]
 def run(flags,expected,swiss,lower):
  subprocess.run(command+flags,stdout=subprocess.DEVNULL,check=True)
  d=json.load(open(site/'data.json'));videos=d['videos'];assert len(videos)==expected
  assert sum(v['stage']=='swiss-4' for v in videos)==swiss
  assert len(d['replayPolicy'].get('archivedGames',[]))==lower
  referenced={u for v in videos for u in [v.get('mp4'),v['poster'],v['chaptersFile']] if u}|{part['url'] for v in videos for part in v.get('parts',[])}
  assert all((site/u).is_file() for u in referenced)
  if lower:
   removed=json.load(open(view/'lower-before-wave4-public-catalog-archived.json'))
   for v in removed:
    for u in [v.get('mp4'),v['poster'],v['chaptersFile']]+[x['url'] for x in v.get('parts',[])]:
     if u:assert not (site/u).exists()
  results.append(dict(flags=flags,games=expected,swiss4=swiss,archivedLowerGames=lower))
 run([],34,14,0)
 run(['--archive-lower-before-wave','4'],26,14,8)
 run([],34,14,0)
 run(['--omit-swiss4'],20,0,0)
import hashlib
verified=0
for stage in ('swiss-4','playoff-1','playoff-2','playoff-3'):
 for pair in json.load(open(base/(stage+'.json')))['draw']:
  for game in pair['games']:
   source=base/game['recording']['path']
   assert hashlib.sha256(source.read_bytes()).hexdigest()==game['recording']['sha256']
   verified+=1
assert verified==34
proof=dict(passed=True,newGames=0,rawFilesChanged=0,sourceRecordingsVerified=verified,cases=results)
(root/'local-evidence/tournament-2026-11-04/archival-flags-verification.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
