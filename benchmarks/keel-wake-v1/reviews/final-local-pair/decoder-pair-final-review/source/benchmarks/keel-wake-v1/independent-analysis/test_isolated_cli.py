"""Actual isolated subprocess command integration; no game/JVM/engine starts."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import evidence_fixture as f

a=f.audit
SCRIPT=Path(a.__file__).resolve()


def execute(command,root,label):
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120,check=False)
    (root/(label+'.stdout')).write_bytes(result.stdout);(root/(label+'.stderr')).write_bytes(result.stderr)
    f.dump(root/(label+'.command.json'),{'command':command,'exitCode':result.returncode,'pythonVersion':sys.version,'scope':'Isolated stdlib evidence reader only, no engine/JVM/network/game'})
    if result.returncode:
        # Preserve synthetic failed inputs/logs under the bootstrap artifact path.
        dest=Path(__file__).resolve().parents[3]/'build/keel-wake-bootstrap/independent-cli-failures'/label
        shutil.copytree(root,dest,dirs_exist_ok=True)
    return result


def command(x,verb,output):
    return [sys.executable,'-I','-B','-S',str(SCRIPT),verb,'--evidence-zip',str(x['zip']),'--zip-sha256',a.digest(x['zip']),'--source-root',str(x['source']),'--provider-evidence',str(x['provider']),'--stage',x['stage'],'--head',x['head'],'--run',str(x['run']),'--output',str(output)]


class IsolatedReaderProcessTests(unittest.TestCase):
    def test_actual_isolated_help_imports_only_trusted_frozen_siblings(self):
        with tempfile.TemporaryDirectory() as t:
            result=execute([sys.executable,'-I','-B','-S',str(SCRIPT),'--help'],Path(t),'help')
            self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'));self.assertIn(b'derive-phase-order',result.stdout)

    def test_actual_isolated_screen_full_wire_and_derivation(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);x=f.full_fixture(root/'fixture',capture_end_second=.5,causal_overlap=True)
            proof=x['provider']/'phase-order-proof.json';proof.unlink()
            result=execute(command(x,'derive-phase-order',proof),root,'screen-derive')
            self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace')+result.stdout.decode(errors='replace'))
            output=root/'screen-output.json';argv=command(x,'audit-stage',output)+['--custody-zip',str(x['custodyZip']),'--custody-sha256',a.digest(x['custodyZip']),'--reviewer-task',x['reviewer'],'--reviewer-identity-file',str(x['identity'])]
            result=execute(argv,root,'screen-audit');self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace')+result.stdout.decode(errors='replace'))
            data=json.loads(output.read_text());self.assertEqual(data['decision'],'PASS');self.assertEqual(data['registeredRoundBudget'],120);self.assertEqual(len(data['validGames']),14)

    def test_actual_isolated_fresh_confirmation_and_original_screen_replay(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);screen=f.full_fixture(root/'screen',capture_end_second=.5,causal_overlap=True);x=f.full_fixture(root/'confirmation',stage='confirmation',run_id=8,job_id=84,capture_end_second=.5,causal_overlap=True)
            gate=f.attach_upstream_screen(screen,x);f.bind_fixture_upstream_inputs(x,gate)
            proof=x['provider']/'phase-order-proof.json';proof.unlink()
            result=execute(command(x,'derive-phase-order',proof),root,'confirmation-derive');self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace')+result.stdout.decode(errors='replace'))
            output=root/'confirmation-output.json';argv=command(x,'audit-stage',output)+['--custody-zip',str(x['custodyZip']),'--custody-sha256',a.digest(x['custodyZip']),'--reviewer-task',x['reviewer'],'--reviewer-identity-file',str(x['identity'])]
            result=execute(argv,root,'confirmation-audit');self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace')+result.stdout.decode(errors='replace'))
            data=json.loads(output.read_text());self.assertEqual(data['decision'],'PASS');self.assertEqual(data['registeredRoundBudget'],300);self.assertEqual(len(data['validGames']),18);self.assertTrue(data['upstreamScreen']['pass'])


if __name__=='__main__':unittest.main()
