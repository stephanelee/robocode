#!/usr/bin/env python3
"""Reproduce the bounded historical audit; no engine, opponent logs or models opened."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile
import analyze

JAR_SHA='817a9d53ce944d27e3f536a92869eeca2e0aec20ef5c52b2dab882e931205792'
PREFIX='build/facet-diamond-repaired-100/'

def main(args):
    if args.jar.stat().st_size>5*1024*1024 or hashlib.sha256(args.jar.read_bytes()).hexdigest()!=JAR_SHA:raise ValueError('historical measured Facet JAR required')
    args.output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='facet-audit-') as name:
        temporary=Path(name)
        if args.parts_root:
            manifest=json.loads((args.parts_root/'artifact-provenance.json').read_text())
            if manifest['artifactSHA256']!='957e78cc5574c6c8619809a06ede7c5d08bc91925281b85aba822fa28546dbb1':raise ValueError('unexpected historical artifact')
            if len(manifest['parts'])!=20:raise ValueError('exact20 archive parts required')
            archive=temporary/'own-evidence.zip';digest=hashlib.sha256();size=0
            with archive.open('wb') as out:
                for index,part in enumerate(manifest['parts']):
                    if part['path']!=f'raw-artifact.zip.part{index:03d}':raise ValueError('unexpected part name/order')
                    path=(args.parts_root/part['path']).resolve()
                    if not path.is_relative_to(args.parts_root.resolve()):raise ValueError('part outside evidence root')
                    if path.stat().st_size>1048576:raise ValueError('part cap')
                    data=path.read_bytes()
                    if len(data)>1048576 or len(data)!=part['bytes'] or hashlib.sha256(data).hexdigest()!=part['sha256']:raise ValueError('part identity')
                    out.write(data);digest.update(data);size+=len(data)
            if size!=20752953 or digest.hexdigest()!=manifest['artifactSHA256']:raise ValueError('archive identity')
            root=temporary/'own'
            # Exact allowlist: no runner.log, private opponent files, or even own model exports.
            members=['registration.json']+[f'diamond-port-g{i:02d}/{leaf}' for i in range(1,6) for leaf in ('FACET_trace.jsonl','FACET_metrics.jsonl')]
            with zipfile.ZipFile(archive) as source:
                for member in members:
                    info=source.getinfo(PREFIX+member)
                    if info.file_size>33554432:raise ValueError('own evidence member cap')
                    target=root/member;target.parent.mkdir(parents=True,exist_ok=True)
                    with source.open(PREFIX+member) as inp,target.open('wb') as out:shutil.copyfileobj(inp,out)
        else:root=args.root
        result=analyze.analyze(root)
        expected=dict(traceRows=500000,submittedDecisions=33615,confirmedPrefixShots=797,independentlyReconciledConfirmations=797)
        for key,value in expected.items():
            if result['totals'].get(key)!=value:raise ValueError('historical replay differs: '+key)
        for key in ('headingPass','originPass','requestNextTickPass','creationSameTickPass','firstMotionSameTickPass','pendingPass','submittedPass'):
            if result['totals'].get(key)!=797:raise ValueError('historical physical invariant differs: '+key)
        (args.output/'trace-analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        java_home=args.java_home
        compiler=str(java_home/'bin/javac');java=str(java_home/'bin/java')
        runtime=subprocess.run([java,'-version'],capture_output=True,text=True,check=True)
        if 'version "17.' not in runtime.stderr+runtime.stdout:raise ValueError('Java17 required')
        env={k:v for k,v in os.environ.items() if not k.startswith(('FACET_','AISURVIVOR_','RETCON_')) and k not in ('JAVA_TOOL_OPTIONS','_JAVA_OPTIONS','JDK_JAVA_OPTIONS')}
        classes=temporary/'classes';classes.mkdir()
        source=Path(__file__).with_name('PreparedPredictionMismatch.java')
        subprocess.run([compiler,'--release','17','-cp',str(args.jar.resolve()),'-d',str(classes),str(source)],check=True,env=env)
        fixture=subprocess.run([java,'-Xmx256m','-cp',str(classes)+os.pathsep+str(args.jar.resolve()),'PreparedPredictionMismatch'],capture_output=True,text=True,check=True,env=env)
        (args.output/'mismatch-output.txt').write_text(fixture.stdout)
        (args.output/'runtime.txt').write_text(runtime.stdout+runtime.stderr)
        print('PASS historical audit:500000ownrows/797physicalconfirmations; populated-store mismatch reproduced; battles=0 training=0 gameplayEdits=0')
        print(fixture.stdout.strip())

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);inputs=parser.add_mutually_exclusive_group(required=True);inputs.add_argument('--root',type=Path);inputs.add_argument('--parts-root',type=Path);parser.add_argument('--jar',type=Path,required=True);parser.add_argument('--java-home',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);main(parser.parse_args())
