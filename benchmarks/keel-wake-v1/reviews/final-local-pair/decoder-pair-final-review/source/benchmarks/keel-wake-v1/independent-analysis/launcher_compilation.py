"""Read-only observer javac25 proof binding. Never invokes Java or a process."""
import math
from pathlib import PurePosixPath

PROOF_KEYS={'schemaVersion','status','command','cwd','source','classpath','compiler','compilerVersionCommand','compilerVersion','stdout','stderr','exitCode','classes','startedAtUTC','completedAtUTC','elapsedMonotonicSeconds'}


def validate(evidence,profile,freeze,read_json,digest,clock):
    def need(ok,label):
        if not ok:raise ValueError(label)
    binding=profile['launcherCompilation']
    need(type(binding) is dict and set(binding)=={'proof','proofSHA256'},'Exact observer compilation profile binding')
    root=evidence/'provenance/launcher-compilation';proof=read_json(root/'proof.json')
    need(type(proof) is dict and set(proof)==PROOF_KEYS and binding['proof']==proof and binding['proofSHA256']==digest(root/'proof.json'),'Whole original javac25 proof bytes')
    need(type(proof['schemaVersion']) is int and proof['schemaVersion']==1 and proof['status']=='valid' and type(proof['exitCode']) is int and proof['exitCode']==0,'Successful source/typecheck only, no launcher execution')
    workspace=profile['identity']['workspace'];home=profile['java25']['home']
    for key in ('source','classpath','compiler'):
        need(type(proof[key]) is dict and set(proof[key])=={'path','sha256'} and type(proof[key]['path']) is str and proof[key]['path'].startswith('/') and '..' not in PurePosixPath(proof[key]['path']).parts,'Exact actual javac source/classpath/compiler reference')
    need(proof['source']['path']==workspace+'/tools/KeelWakeBattle.java' and proof['source']['sha256']==freeze['assets']['tools/KeelWakeBattle.java'],'Actual frozen source launcher typecheck')
    need(proof['classpath']['sha256']==profile['runner'],'Exact pinned runner/API classpath bytes')
    need(proof['compiler']['path']==home+'/bin/javac' and proof['compiler']['sha256']==profile['java25']['files']['bin/javac']['sha256'],'Compiler is exact prepared Java25 graph binary')
    need(proof['compilerVersionCommand']==[home+'/bin/javac','-version'] and type(proof['compilerVersion']) is str and proof['compilerVersion'].strip()=='javac 25.0.5','Actual exact javac25 version output')
    need(type(proof['cwd']) is str and (proof['cwd']==workspace or proof['cwd'].startswith(workspace+'/')) and '..' not in PurePosixPath(proof['cwd']).parts,'Actual owned compilation cwd')
    command=proof['command'];need(type(command) is list and len(command)==7 and type(command[5]) is str and command[5].endswith('/launcher-compilation/classes'),'Exact absolute compiler class output role')
    stage_root=command[5][:-len('/launcher-compilation/classes')];need(stage_root.startswith(workspace+'/') and '..' not in PurePosixPath(stage_root).parts,'Actual stage-owned compilation output')
    expected=[home+'/bin/javac','-proc:none','-cp',proof['classpath']['path'],'-d',stage_root+'/launcher-compilation/classes',proof['source']['path']]
    need(proof['command']==expected,'Exact bounded javac25 -proc:none source/typecheck command, never main')
    for key in ('stdout','stderr'):
        ref=proof[key];need(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and ref['path']==stage_root+'/launcher-compilation/'+key+'.bin' and type(ref['bytes']) is int and ref['bytes']>=0,'Exact original compiler output ref')
        actual=root/(key+'.bin');need(actual.is_file() and actual.stat().st_size==ref['bytes'] and digest(actual)==ref['sha256'],'Whole original javac diagnostic bytes')
    classes=proof['classes'];need(type(classes) is dict and set(classes)=={'KeelWakeBattle.class'},'Exact observer-only compiled class inventory')
    for name,ref in classes.items():
        need(type(ref) is dict and set(ref)=={'sha256','mode'} and type(ref['mode']) is int and 0<=ref['mode']<=0o7777,'Exact compiled file hash/mode')
        path=root/'classes'/name;need(path.is_file() and digest(path)==ref['sha256'] and path.stat().st_mode&0o7777==ref['mode'],'Whole actual source/typecheck class bytes/mode')
    actual={p.name for p in (root/'classes').glob('*')}
    need(actual==set(classes),'No omitted/additional compiler outputs')
    elapsed=proof['elapsedMonotonicSeconds'];need(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<=120,'Complete version+typecheck bounded120seconds')
    start,end=clock(proof['startedAtUTC']),clock(proof['completedAtUTC']);need(0<=(end-start).total_seconds()<=120,'Exact compiler proof UTC interval')
    return True
