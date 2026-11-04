#!/usr/bin/env python3
"""Bounded read-only audit of own Facet chronology. No battles, models or training."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

MAX_LINES=100000

def number(value,key):
    found=re.search(r'(?<![A-Za-z])'+re.escape(key)+r'=([-+0-9.eE]+)(?= |,|$)',value)
    if not found:raise ValueError('missing numeric field '+key)
    return float(found.group(1))

def wrap(angle):return (angle+math.pi)%(2*math.pi)-math.pi

def analyze(root):
    root=root.resolve();registration=json.loads((root/'registration.json').read_text())
    assert registration['facetSourceSHA256']=='dc5cc37c364b7ff76b122116cdc6a3d65a85abd102202a2f75f8add65b68ec5d'
    all_games=[];total=Counter();per_gun=Counter();rating_checks=Counter()
    for index in range(1,6):
        directory=root/f'diamond-port-g{index:02d}';trace=directory/'FACET_trace.jsonl'
        if trace.stat().st_size>33554432 or (directory/'FACET_metrics.jsonl').stat().st_size>2097152:raise ValueError('own audit file cap')
        kinds=Counter();shots={};states={};plans={};submitted={};outcomes={};motions={};native_fired=set();all_rating_rounds={};counts=Counter();guns=Counter();examples=[];last=None
        with trace.open() as stream:
            for line_number,line in enumerate(stream,1):
                if line_number>MAX_LINES:raise ValueError('trace exceeds declared audit bound')
                row=json.loads(line);kind=row['kind'];value=row['value'];key=(row['round'],row['turn']);kinds[kind]+=1;last=key
                if kind=='own.state':
                    states[key]={k:number(value,k) for k in ('x','y','gun','heat','energy')}
                elif kind=='fire.prepared':
                    plans[key]=dict(angle=number(value,'angle'),power=number(value,'power'),gun=re.search(r'gun=(\w+)',value).group(1))
                elif kind=='command.submitted':
                    submitted[key]=row
                    state=states.get(key)
                    if state and state['heat']==0 and state['energy']>.3:
                        counts['coolFundedDecisions']+=1
                        old=plans.get((key[0],key[1]-1))
                        if old:
                            counts['coolFundedWithPriorPlan']+=1
                            error=abs(wrap(state['gun']-old['angle']))
                            if state['energy']>old['power'] and error<=math.radians(.5)+1e-9:
                                counts['coolPriorPlanFundedAlignedWithinMaxTolerance']+=1
                                if 'fire=false' in value:
                                    counts['coolPriorPlanFundedAlignedButNoSubmission']+=1
                                    if len(examples)<3:examples.append(dict(line=line_number,round=key[0],turn=key[1],gun=old['gun'],angleErrorDegrees=math.degrees(error),energy=state['energy'],power=old['power']))
                    if 'fire=true' in value:counts['submittedFireCommands']+=1
                elif kind=='BulletFiredEvent':
                    native_fired.add((key[0],int(value.split(' callbackSequence=')[0]),key[1]))
                elif kind=='bullet.motion':
                    motions.setdefault((key[0],int(number(value,'id'))),key[1])
                elif kind=='fire.confirmed':
                    bullet=int(number(value,'id'));request=int(number(value,'request'));requested=(key[0],request);gun=re.search(r'gun=(\w+)',value).group(1)
                    if (key[0],bullet) in shots:raise ValueError('duplicate own confirmation')
                    counts['confirmedPrefixShots']+=1;guns[gun]+=1
                    counts['nativeFiredRecordPass' if (key[0],bullet,key[1]) in native_fired else 'nativeFiredRecordMissing']+=1
                    state=states.get(requested);command=submitted.get(requested)
                    if state and command:
                        heading=math.radians(number(value,'heading'));pos=re.search(r'position=([-+0-9.eE]+),([-+0-9.eE]+)',value);error=abs(wrap(heading-state['gun']));distance=math.hypot(float(pos.group(1))-state['x'],float(pos.group(2))-state['y'])
                        checks=dict(heading=error<=1e-7,origin=distance<=1e-6,requestNextTick=key[1]==request+1,creationSameTick=int(number(value,'creation'))==key[1],firstMotionSameTick=int(number(value,'firstMotion'))==key[1],pending='pending=true' in value,submitted='fire=true' in command['value'])
                        counts['independentlyReconciledConfirmations']+=1
                        for name,valid in checks.items():counts[name+'Pass' if valid else name+'Mismatch']+=1
                    else:counts['confirmationMissingRequestEvidence']+=1
                    shots[(key[0],bullet)]=dict(gun=gun,round=key[0],request=request,confirmation=key[1])
                elif kind in ('BulletHitBotEvent','BulletHitWallEvent','BulletHitBulletEvent'):
                    identity=value.split(' callbackSequence=')[0]
                    parts=identity.split(':')
                    candidates={int(parts[1])} if kind=='BulletHitBotEvent' else {int(parts[0])}
                    if kind=='BulletHitBulletEvent':candidates.add(int(parts[2]))
                    for bullet in candidates:
                        shot_key=(key[0],bullet)
                        if shot_key in shots:
                            if shot_key in outcomes:counts['duplicateOutcomeRecords']+=1
                            outcomes[shot_key]=kind
                elif kind=='gun.rating':
                    gun=re.search(r'gun=(\w+)',value).group(1);cohort=int(number(value,'cohort'));raw=number(value,'raw');adjusted=number(value,'adjusted')
                    all_rating_rounds.setdefault(key[0],{})[gun]=dict(cohort=cohort,raw=raw,adjusted=adjusted)
                    if not 0<=raw<=1 or not 0<=adjusted<=10:raise ValueError('rating outside construction bounds')
        metric_rows=[json.loads(x) for x in (directory/'FACET_metrics.jsonl').read_text().splitlines()]
        final=metric_rows[-1]
        if line_number!=MAX_LINES:raise ValueError('unexpected acquired trace coverage')
        complete_rounds=[]
        for round_id,views in sorted(all_rating_rounds.items()):
            if set(views)=={'analytic','long','recent'}:
                rating_checks['completeCommonRatingRoundSnapshots']+=1
                if len({v['cohort'] for v in views.values()})!=1:raise ValueError('unequal common-cohort denominators')
                complete_rounds.append(round_id)
        outcome_by_gun={}
        for key,shot in shots.items():
            motion=motions.get(key)
            counts['firstObservedMotionMissing' if motion is None else 'firstObservedMotionSameTick' if motion==shot['confirmation'] else 'firstObservedMotionLater']+=1
            result=outcomes.get(key,'not-observed-before-cap')
            outcome_by_gun.setdefault(shot['gun'],Counter())[result]+=1
        counts['traceRows']=line_number;counts['submittedDecisions']=kinds['command.submitted'];counts['prefixHitCallbacks']=kinds['BulletHitBotEvent']
        total.update(counts);per_gun.update(guns)
        all_games.append(dict(game=index,traceSHA256=hashlib.sha256(trace.read_bytes()).hexdigest(),traceRows=line_number,lastObservedIdentity=dict(round=last[0],turn=last[1]),recordKinds=dict(kinds),checks=dict(counts),gunConfirmations=dict(guns),observedOutcomesByGun={k:dict(v) for k,v in outcome_by_gun.items()},commonRatingRoundSnapshots=all_rating_rounds,mechanicallyAlignedHoldExamples=examples,fullGameConfirmed=final['shotsConfirmed'],fullGameHitCallbacks=final['counters'].get('shot.hit',0)))
    return dict(schema=1,purpose='offline gun/wave audit; no battles/training/parameter edits',sourceSHA256=registration['facetSourceSHA256'],jarSHA256=registration['facetJarSHA256'],configSHA256=registration['facetConfigSHA256'],inputBudget='five own100000-line capped traces and100ownmetricrows; no models or opponent logs',totals=dict(total),confirmedByGunInCoveredPrefix=dict(per_gun),ratingChecks=dict(rating_checks),games=all_games,limitations=['Unobserved post-cap outcomes are not misses. Prefix gun outcomes are selected/nonrandom and do not compare competitive gun strength.','Actual bullet power and immutable per-gun wave predictions/labels are not independently serialized in these traces. pending=true is bot validation evidence, not independent actual-power replay.','Cool/funded/aligned does not imply every plan gate passes; For this registered800x600 field the plan tolerance is capped at0.5degrees; drift/freshness/policy gates remain. No rejection reasons were logged.','No model read, fit, search, parameter or gameplay modification and no new engine game.'])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=analyze(args.root);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(totals=result['totals'],guns=result['confirmedByGunInCoveredPrefix'],ratings=result['ratingChecks']),indent=2))
