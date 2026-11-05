#!/usr/bin/env python3
"""One immutable old baseline game's parser validation; NEVER new Brace evidence."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--zip',type=Path,required=True)
parser.add_argument('--prior-audit',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args()
source=Path(__file__).resolve().parents[1]/'audit.py'
spec=importlib.util.spec_from_file_location('independent_brace_audit',source)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
expected='bcbd2e27ff64dc585f7341aa32ae506ea9819903c1893db8a723ba91c7036ad6'
assert audit.digest(a.zip)==expected
prior=audit.read_json(a.prior_audit)
assert prior['stage']=='confirmation' and prior['expectedRun']==41889609469
store=audit.ZipStore(a.zip)
try:
    root=store.root('robocode/robocode/build/keel-helm/confirmation/games/c01')
    recordings=root.glob('*.battle.gz');assert len(recordings)==1
    current=audit.reduce_game(recordings[0],root/'markers.txt','Keel','Diamond Port','1.8.28',20,False)
    assert current['valid'],current
    old=next(r for r in prior['validGames'] if r['game']['id']=='c01')
    fields=list(audit.COUNTS)+['recordingSHA256','markerSHA256','ownScore','totalScore','wins','rounds','maxMs','skips','timingPass','turns','hitRate','enemyHitRate','firingPer1000Turns','exactShare','powerMix','enemyPowerMix','perRound']
    differences=[k for k in fields if current[k]!=old[k]];assert not differences,differences
    result=dict(scope='HISTORICAL PARSER-FORMAT VALIDATION ONLY; zero new Brace games or strength evidence; no candidate/harness imports',originalZIP=str(a.zip),originalZIPBytes=a.zip.stat().st_size,originalZIPExpectedSHA256=expected,originalZIPActualSHA256=audit.digest(a.zip),recordingMember=recordings[0].member,recordingSHA256=current['recordingSHA256'],historicalRegisteredGame='KeelHelm confirmation baseline c01',historicalRounds=20,newBraceRounds=0,fieldsCompared=fields,mismatches=differences,**{'pass':True},python=sys.version,auditorSourceSHA256=audit.digest(source),priorIndependentAuditSHA256=audit.digest(a.prior_audit))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(scope=result['scope'],fieldsCompared=len(fields),mismatches=len(differences))))
finally:store.close()
