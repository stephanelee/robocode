"""Read-only, predeclared exploratory summaries of the spent PR63 legal defense trace."""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path

EXPECTED="2c42ba7b3ed7a761565ed65016354918749fccda7de0884ac746926c5008979d"

def stats(values):
    values=sorted(float(x) for x in values if isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x))
    return {"n":len(values),"min":min(values) if values else None,"mean":sum(values)/len(values) if values else None,
            "median":(values[(len(values)-1)//2]+values[len(values)//2])/2 if values else None,
            "p90":values[math.ceil(.9*len(values))-1] if values else None,"max":max(values) if values else None}

def relative(a):return (a+180)%360-180
def speed(w):return 20-3*w["power"]
def key(w):return w["round"],w["owner"],w["creationTurn"]
def same_slot(slot,w):
    return slot[0]==w["creationTurn"] and abs(slot[1]-w["originX"])<=1e-6 \
        and abs(slot[2]-w["originY"])<=1e-6 and abs(slot[3]-speed(w))<=1e-6

def analyze(rows):
    waves=[(i,r) for i,r in enumerate(rows) if r["type"]=="wave"]
    hits=[(i,r) for i,r in enumerate(rows) if r["type"]=="enemyHit"]
    planners=[(i,r) for i,r in enumerate(rows) if r["type"]=="planner"]
    wave_keys=collections.defaultdict(list)
    for i,w in waves:wave_keys[key(w)].append((i,w))
    chosen_counts=collections.Counter();profile_counts=collections.Counter();unsafe_counts=collections.Counter()
    margins=[];risks=[];support=[];first_contacts=[];done_times=[];wall_times=[];stop_advantage=[]
    for _,p in planners:
        scores=p["scoresCCW_STOP_CW"];choice=p["chosen"]+1;chosen_counts[str(p["chosen"])]+=1
        profiles=sum(bool(w[7]) for w in p["waves"]);profile_counts[str(profiles)]+=1
        support.extend(w[8] for w in p["waves"])
        finite=[x for x in scores if x is not None];unsafe_counts[str(3-len(finite))]+=1
        if scores[choice] is not None:
            risks.append(scores[choice]);margins.append(scores[choice]-min(finite))
            others=[s for j,s in enumerate(scores) if j!=choice and s is not None]
            if others:stop_advantage.append(min(others)-scores[choice])
        timing=p["rolloutTimesCCW_STOP_CW"][choice]
        first_contacts.extend(x for x in timing[:-2:2] if x>=0)
        done_times.extend(x for x in timing[1:-2:2] if x>=0)
        if timing[-2]>=0:wall_times.append(timing[-2])
    results=[]
    for index,hit in hits:
        result={"round":hit["round"],"turn":hit["turn"],"owner":hit["owner"],"matched":hit.get("matched"),
                "creationTurn":hit.get("creationTurn"),"radialResidual":hit.get("radialResidual")}
        candidates=[w for i,w in wave_keys.get((hit["round"],hit["owner"],hit.get("creationTurn")),[]) if i<index]
        if not hit.get("matched"):result["waveJoin"]="unmatched"
        elif len(candidates)!=1:result["waveJoin"]="missing" if not candidates else "ambiguous"
        else:
            wave=candidates[0];result["waveJoin"]="unique";v=speed(wave)
            result["residualBulletTurns"]=hit["radialResidual"]/v
            escape=math.degrees(math.asin(8/v));gf=relative(hit["direction"]-wave["referenceBearing"])*wave["sign"]/escape
            own=wave.get("aimOwn")
            if own is not None:
                lateral=abs(own[3]*math.sin(math.radians(own[2]-wave["referenceBearing"])))
                linear=math.asin(min(1,lateral/v))/math.asin(8/v)
                result.update(context=wave["context"],actualHitGF=gf,priorLinearGF=linear,
                              nearestPriorModeGFDistance=min(abs(gf),abs(gf-linear)))
            prior=[(i,p) for i,p in planners if i<index and p["round"]==hit["round"] and p["turn"]<hit["turn"]]
            if not prior:result["plannerJoin"]="missing"
            else:
                pi,p=max(prior,key=lambda item:(item[1]["turn"],item[0]));choice=p["chosen"]+1
                slots=[];ambiguous=False
                for slot_index,slot in enumerate(p["waves"]):
                    identities=[w for wi,w in waves if wi<pi and w["round"]==p["round"] and same_slot(slot,w)]
                    if len(identities)!=1:ambiguous=True
                    elif key(identities[0])==key(wave):slots.append(slot_index)
                result.update(plannerJoin="unique",plannerTurn=p["turn"],decisionLag=hit["turn"]-p["turn"],
                              chosen=p["chosen"],scores=p["scoresCCW_STOP_CW"],
                              unsafe=[p["unsafeCCW"],p["unsafeSTOP"],p["unsafeCW"]],
                              timings=p["rolloutTimesCCW_STOP_CW"],selectedWaves=p["waves"],
                              anyProfile=any(w[7] for w in p["waves"]),matchedWaveSlot=None)
                if len(slots)==1:
                    s=slots[0];slot=p["waves"][s];timing=p["rolloutTimesCCW_STOP_CW"][choice]
                    result.update(matchedWaveSlot=s,matchedWaveProfile=slot[7],matchedWaveSupport=slot[8],
                                  matchedWaveTTC=slot[4],chosenFirstContact=timing[2*s],chosenDone=timing[2*s+1])
                else:
                    available=any(wi<pi and key(w)==key(wave) for wi,w in waves)
                    result["waveInferredBeforeDecision"]=available
                    result["slotMissingReason"]="not_yet_inferred" if not available else "ambiguous_identity" if ambiguous else "not_selected_or_no_longer_active"
        results.append(result)
    joins=collections.Counter(r["waveJoin"] for r in results)
    coverage=collections.Counter("selected" if r.get("matchedWaveSlot") is not None else r.get("slotMissingReason",r.get("plannerJoin","wave_unknown")) for r in results)
    profiles=collections.Counter(str(r.get("matchedWaveProfile","unknown")) for r in results)
    return {"eventCounts":dict(collections.Counter(r["type"] for r in rows)),"waveJoin":dict(joins),
            "inferredWithoutObservedHitLabel":len(waves)-len({(r["round"],r["owner"],r["creationTurn"]) for r in results if r["waveJoin"]=="unique"}),
            "radialResidualPixels":stats(r.get("radialResidual") for r in results),
            "radialResidualBulletTurns":stats(r.get("residualBulletTurns") for r in results),
            "preHitCoverage":dict(coverage),"preHitDecisionLag":stats(r.get("decisionLag") for r in results),
            "matchedWaveProfileAtPreHit":dict(profiles),"matchedWaveSupportAtPreHit":stats(r.get("matchedWaveSupport") for r in results),
            "actualHitGF":stats(r.get("actualHitGF") for r in results),
            "nearestFixedPriorModeGFDistance":stats(r.get("nearestPriorModeGFDistance") for r in results),
            "plannerSummary":{"decisions":len(planners),"choices":dict(chosen_counts),"profileSlots":dict(profile_counts),
                "unsafeOptions":dict(unsafe_counts),"supportMass":stats(support),"chosenRisk":stats(risks),
                "chosenMinusFiniteMinimum":stats(margins),"bestOtherMinusChosen":stats(stop_advantage),
                "chosenFirstContactSteps":stats(first_contacts),"chosenDoneSteps":stats(done_times),"chosenWallVetoSteps":stats(wall_times)},
            "hits":results,
            "limits":["Spent five-round exploratory trace, not fresh validation or an intervention.",
                      "Radial matches are legal inferred associations, not independently verified opponent bullet origins/targets.",
                      "No hit label on an inferred wave is unknown, not successful evasion or false inference.",
                      "Recorded alternate scores/contact times describe unexecuted held policies; no alternate survival claim.",
                      "Prior-mode alignment is not full smoothed-density risk: per-decision bin/footprint arrays were not recorded.",
                      "The prior includes uniform mass; distance from its peaks does not imply zero danger or prove primary cause.",
                      "One-step own physics parity does not validate long rollouts, waves or danger calibration."]}

def main():
    p=argparse.ArgumentParser();p.add_argument("trace",type=Path);p.add_argument("output",type=Path);args=p.parse_args()
    data=gzip.decompress(args.trace.read_bytes());digest=hashlib.sha256(data).hexdigest()
    if digest!=EXPECTED:raise ValueError("not the frozen PR63 trace")
    rows=[json.loads(line) for line in data.splitlines() if line.strip()]
    result=analyze(rows);result["traceSHA256"]=digest
    result["traceGzipSHA256"]=hashlib.sha256(args.trace.read_bytes()).hexdigest()
    result["status"]=json.loads(Path(str(args.trace.with_suffix(""))+".status.json").read_text())
    result["roundSummaries"]=[r for r in rows if r["type"]=="round"]
    result["analysisSHA256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["planSHA256"]=hashlib.sha256(Path(__file__).with_name("analysis-plan.md").read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")

if __name__=="__main__":main()
