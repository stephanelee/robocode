"""Opaque archival fixtures only. No bot/server/recording metrics are executed."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

spec=importlib.util.spec_from_file_location('private_packager','/private/tmp/keel-helm-confirmation-packager.py')
pkg=importlib.util.module_from_spec(spec);spec.loader.exec_module(pkg)


class Packaging(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='keel-confirmation-archive-test-',dir='/private/tmp')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def args(self,members,kind='evidence'):
        z=self.root/'original.zip'
        with zipfile.ZipFile(z,'w',compression=zipfile.ZIP_STORED) as f:
            for name,raw in members:f.writestr(name,raw)
        raw=z.read_bytes()
        return argparse.Namespace(zip=z,expected_zip_sha256=hashlib.sha256(raw).hexdigest(),
            expected_zip_bytes=len(raw),output=self.root/'archive',kind=kind,stage='confirmation',
            stage_prefix=None,run_id=41889609469,expected_head='f2a25d3309062a362a13902329de405bf63094a7',
            extras_manifest=None,extra=[])

    def verify(self,args):
        subprocess.run([sys.executable,'-I','-B','-S',str(args.output/'verify.py')],check=True,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=10)

    def test_partial_inventory_preserves_one_whole_opaque_recording_and_unplayed(self):
        raw=b'opaque archival fixture: not a game recording\n'*(3*1024*1024//44)
        decision=b'{"decision":"invalid","unplayedIDs":["c02","c03"]}\n'
        a=self.args([('root/build/keel-helm/confirmation/games/c01/fixture.battle.gz',raw),
            ('root/build/keel-helm/confirmation/decision.json',decision),
            ('root/build/keel-helm/confirmation/work/c01/bot/input.txt',b'original staged input')])
        result=pkg.build(a);self.assertEqual(result['wholeRecordings'],1)
        self.assertEqual((a.output/'stage/decision.json').read_bytes(),decision)
        self.assertEqual((a.output/'stage/games/c01/fixture.battle.gz').read_bytes(),raw)
        m=json.loads((a.output/'manifest.json').read_bytes())
        self.assertTrue(m['archiveWarnings']);self.assertGreater(len(m['originalZIP']['parts']),1)
        self.assertTrue(all((a.output/p).stat().st_size<=2*1024*1024 for p in m['originalZIP']['parts']))
        self.verify(a)
        before=(a.output/'manifest.json').read_bytes()
        with self.assertRaises(FileExistsError):pkg.build(a)
        self.assertEqual((a.output/'manifest.json').read_bytes(),before)

    def test_failed_preparation_has_zero_recordings_without_fabricated_decision(self):
        failure=b'{"phase":"prepare","noGamesStarted":true}\n'
        a=self.args([('root/build/keel-helm/confirmation/failure-prepare.json',failure),('build.log',b'build failed\n')])
        result=pkg.build(a);self.assertEqual(result['wholeRecordings'],0);self.assertEqual(result['decisionFiles'],0)
        self.assertEqual(result['failureArtifacts'],1);self.verify(a)

    def test_provenance_flat_source_bytes_remain_in_exact_original_zip(self):
        a=self.args([('profile.json',b'{"identity":"fixture only"}\n'),('source/input.java',b'not executed fixture')],kind='provenance')
        pkg.build(a);self.assertEqual((a.output/'outside-stage-originals/profile.json').read_bytes(),b'{"identity":"fixture only"}\n');self.verify(a)

    def test_hash_mismatch_and_git_worktree_outputs_fail_before_writes(self):
        a=self.args([('failure.json',b'failure fixture')]);a.expected_zip_sha256='0'*64
        with self.assertRaises(ValueError):pkg.build(a)
        self.assertFalse(a.output.exists())
        a.expected_zip_sha256=hashlib.sha256(a.zip.read_bytes()).hexdigest()
        (self.root/'.git').write_text('fake worktree protection fixture')
        with self.assertRaises(ValueError):pkg.build(a)
        self.assertFalse(a.output.exists())

    def test_stable_extra_hash_validation_and_other_original_zip_parts(self):
        a=self.args([('build.log',b'preparation fixture')]);extra=self.root/'review.json';extra.write_bytes(b'{"review":"fixture"}')
        other=self.root/'pregame.zip';other.write_bytes(b'opaque other provider archive fixture')
        index=self.root/'extras.json';index.write_text(json.dumps({'files':{
            str(extra):dict(bytes=extra.stat().st_size,sha256=hashlib.sha256(extra.read_bytes()).hexdigest()),
            str(other):dict(bytes=other.stat().st_size,sha256=hashlib.sha256(other.read_bytes()).hexdigest())}}))
        a.extras_manifest=index;pkg.build(a);self.verify(a)

    def test_unsafe_original_filename_is_preserved_opaque_never_extracted(self):
        a=self.args([('../unsafe.txt',b'opaque hostile path fixture'),('build.log',b'log')])
        pkg.build(a);self.assertFalse((self.root/'unsafe.txt').exists());self.verify(a)


if __name__=='__main__':unittest.main()
