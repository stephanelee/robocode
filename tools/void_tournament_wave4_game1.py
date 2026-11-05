"""Apply only the user's explicit void/replacement ruling; starts no games."""
from pathlib import Path
import argparse,json,hashlib,datetime
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
base=Path('benchmarks/tournament-2026-11-04');state=base/'playoff-4.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(state)=='4333f3a14a4140773da43dfa714b421ad2359638fa6318703c9afa61865e1ec7'
r=json.loads(state.read_text());pair=r['draw'][0];game=pair['games'][0]
assert r['status']=='incomplete' and r['completedGames']==0 and r['completedRounds']==0 and r['plannedGames']==4
assert pair['matchId']=='U7' and game['game']==1 and game['status']=='failed' and game['participants']==['KeelHelm','Keel']
assert len(pair['games'])==1 and not r['draw'][1]['games'] and len(r['failures'])==1
assert game['directory']=='playoff-4/p01-g1' and game['error']=='CRC check failed 0x232c7523 != 0xee7e5687'
run=base/game['directory'];snapshot=base/'playoff-4-recording-stop-1.json';ruling=base/'playoff-4-void-ruling-1.json';target=base/'playoff-4/voided-attempts/p01-g1-recording-corrupt-1'
assert not snapshot.exists() and not ruling.exists() and not target.exists()
assert sha(run/'battle.json')=='04e88be98f7ff268240b25fb1915430b888f769346247a40a93e41ff5e4bcb85'
recording=next((run/'recordings').glob('*.battle.gz'));assert sha(recording)=='f50ecbd2c25a4df38e0d64429a7ce949fc971033d771c3129f3d4892f45cc8fc'
custody=json.loads(Path('local-evidence/tournament-2026-11-04/playoff-4-stop-custody.json').read_text())
assert all(sha(run/name)==v['sha256'] and (run/name).stat().st_size==v['sizeBytes'] for name,v in custody.items())
record=dict(decision='void-original-and-authorize-one-replacement',authorization='User explicitly selected: Void Game 1 and play a replacement after restoring disk space',scope='Only playoff-4/U7/game1',originalReportSHA256=sha(state),originalOfficialSHA256=sha(run/'battle.json'),originalRecordingSHA256=sha(recording),reason='Substantive recording corruption; no intact complete game replay',preservedDirectory=str(target.relative_to(base)),excludedEngineGames=1,excludedEngineRounds=20,replacementGamesAuthorized=1,replacementLaunchOrder=game['participants'],rawFiles=custody,voidedAt=datetime.datetime.now(datetime.timezone.utc).isoformat())
if not args.apply:print('PASS dry-run: exact failed state, original official/recording and all raw hashes verified; no files changed');raise SystemExit
snapshot.write_bytes(state.read_bytes());ruling.write_text(json.dumps(record,indent=2)+'\n');target.parent.mkdir(parents=True,exist_ok=True);run.rename(target)
r.setdefault('voidedGames',[]).append(dict(**game,preservedDirectory=record['preservedDirectory'],ruling=ruling.name,rulingSHA256=sha(ruling),excludedEngineRounds=20))
r.setdefault('resolvedFailures',[]).append(dict(**r['failures'][0],decision=record['decision'],ruling=ruling.name,rulingSHA256=sha(ruling)))
r['failures']=[];pair['games']=[];r['voidedEngineGames']=1;r['voidedEngineRounds']=20
p=state.with_suffix('.tmp');p.write_text(json.dumps(r,indent=2)+'\n');p.replace(state)
assert all(sha(target/name)==v['sha256'] for name,v in custody.items())
print('Applied user-authorized void; original files preserved byte-for-byte; only game1 reopened; no games launched')
