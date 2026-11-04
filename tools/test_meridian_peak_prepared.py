"""Owned-environment repeat verification guards. Synthetic filesystem fixtures; no games/install."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import benchmark
import meridian_peak_profile as profile
import meridian_peak_verify as verify
import retcon


class PreparedEnvironmentTests(unittest.TestCase):
    def test_one_fresh_collection_then_repeated_verification_never_recreates_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reg=root/'benchmarks/meridian-peak-v1';reg.mkdir(parents=True)
            provenance=root/'build/meridian-peak/provenance';cache=root/'cache';cache.mkdir()
            pinned=profile.REG.parent/'meridian-gun-plan/package-contents.json'
            expected=root/'benchmarks/meridian-gun-plan/package-contents.json';expected.parent.mkdir();shutil.copy2(pinned,expected)
            packages=json.loads(pinned.read_text());walls={'version':'synthetic','files':{}}
            runtime_file=root/'runtime.bin';runtime_file.write_bytes(b'synthetic runtime')
            source=root/'candidate.java';source.write_text('// synthetic candidate')
            identity={'boot':'one','run':'123'}
            def runtime_now():return {'binarySHA':benchmark.digest(runtime_file)}
            original_runtime=runtime_now()
            def fresh_native(*args):
                (root/'fresh-owned-venv').mkdir()  # Like native.prepare, a second creation is an error.
            def initial_profile():
                verify.native.prepare(cache,'unused-fixture-python')
                target=root/'build/meridian/profile.json';target.parent.mkdir(parents=True)
                target.write_text(json.dumps(dict(original_runtime,assets={})))
            def host(stage):
                return dict(stage=stage,bootID=identity['boot'],runID=identity['run'],workspace=str(root),
                    profileSHA256=benchmark.digest(provenance/'profile.json'),
                    preparationSHA256=benchmark.digest(provenance/'prepared-environment.json'),
                    packageSHA256=benchmark.digest(provenance/'package-contents.json'))
            def assets():return {'candidate.java':benchmark.digest(source)}
            patches=(patch.object(profile,'ROOT',root),patch.object(profile,'REG',reg),patch.object(profile,'CACHE',cache),
                patch.object(profile,'PROVENANCE',provenance),patch.object(profile,'allowed_runtime',return_value=[original_runtime]),
                patch.object(profile.original_profile,'main',side_effect=initial_profile),
                patch.object(verify.native,'prepare',side_effect=fresh_native),
                patch.object(profile,'package_contents',return_value=packages),
                patch.object(profile.prepared,'package_contents',return_value=packages),
                patch.object(profile.prepared,'creation_proof',return_value={'schemaVersion':1}),
                patch.object(profile.prepared,'runtime',side_effect=lambda proof:(runtime_now(),root/'fresh-owned-venv/bin/python')),
                patch.object(profile.benchmark,'download',return_value=cache/'synthetic.zip'),
                patch.object(profile,'walls_manifest',return_value=walls),patch.object(profile,'participant_assets',side_effect=assets),
                patch.object(profile,'host_identity',side_effect=host))
            import contextlib
            with contextlib.ExitStack() as stack:
                active=[stack.enter_context(p) for p in patches]
                profile.preflight('screen',fresh=True)
                before={p.name:p.read_bytes() for p in provenance.iterdir()}
                profile.preflight('screen');profile.preflight('screen')
                self.assertEqual(before,{p.name:p.read_bytes() for p in provenance.iterdir()},'Verification must not rewrite preparation evidence')
                active[5].assert_called_once();active[6].assert_called_once()
                with self.assertRaises(ValueError):profile.preflight('screen',fresh=True)
                active[6].assert_called_once()
                identity['boot']='another'
                with self.assertRaisesRegex(ValueError,'binding changed'):profile.preflight('screen')
                identity['boot']='one';identity['run']='124'
                with self.assertRaisesRegex(ValueError,'binding changed'):profile.preflight('screen')
                identity['run']='123'
                runtime_file.write_bytes(b'changed runtime binary')
                with self.assertRaisesRegex(ValueError,'runtime changed'):profile.preflight('screen')
                runtime_file.write_bytes(b'synthetic runtime')
                source.write_text('// changed candidate')
                with self.assertRaisesRegex(ValueError,'participant'):profile.preflight('screen')
                source.write_text('// synthetic candidate')
                with patch.object(profile.prepared,'package_contents',return_value={}):
                    with self.assertRaisesRegex(ValueError,'package contents'):profile.preflight('screen')
                stored=(provenance/'profile.json').read_bytes()
                (provenance/'profile.json').write_bytes(stored+b' ')
                with self.assertRaisesRegex(ValueError,'binding changed'):profile.preflight('screen')

    def test_readonly_native_verifier_checks_real_fixture_files_and_isolated_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp).resolve();base=cache/'strong-python/synthetic';source=base/'source';source.mkdir(parents=True)
            descriptor=source/'bot.json';descriptor.write_text(json.dumps(dict(name='Diamond Port',version='1.8.28',gameTypes=['melee'])))
            wheel=base/'wheels/fixture.whl';wheel.parent.mkdir();wheel.write_bytes(b'synthetic wheel')
            venv=base/'venv';(venv/'bin').mkdir(parents=True)
            python=venv/'bin/python';python.write_bytes(b'synthetic interpreter')
            config=venv/'pyvenv.cfg';config.write_text('include-system-site-packages = false\n')
            (venv/'lib').mkdir();(venv/'lib64').symlink_to('lib',target_is_directory=True)
            entry=dict(path='bot.json',size=descriptor.stat().st_size,sha256=benchmark.digest(descriptor))
            lock=dict(repository='synthetic',commit='synthetic',files=[entry],
                sourceSHA256=hashlib.sha256(('bot.json\0'+entry['sha256']+'\n').encode()).hexdigest(),
                opponents=[dict(sourcePath='',descriptor='bot.json',name='Diamond Port',version='1.8.28')],
                runtime=dict(wheels=[dict(filename=wheel.name,size=wheel.stat().st_size,sha256=benchmark.digest(wheel))]))
            lock_path=cache/'lock.json';lock_path.write_text(json.dumps(lock))
            proof=dict(venv=str(venv.resolve()),pythonExecutable=str(python),pyvenvConfigSHA256=benchmark.digest(config),
                venvFiles=verify.venv_inventory(venv),venvInventoryPolicy=verify.VENV_INVENTORY_POLICY)
            with patch.object(verify,'CACHE',cache),patch.object(verify.native,'manifest',return_value=lock),patch.object(verify.native,'LOCK',lock_path), \
                 patch.object(verify.subprocess,'check_output',return_value='{"version":"synthetic","packages":[]}') as metadata, \
                 patch.object(verify.subprocess,'run',return_value=types.SimpleNamespace(stdout='No broken requirements found.')) as pip, \
                 patch.object(verify.native,'prepare',side_effect=AssertionError('must never prepare')), \
                 patch.object(verify.retcon,'read_url',side_effect=AssertionError('must never download')):
                executable,_=verify.verify_native(proof);self.assertEqual(python,executable)
                self.assertEqual(['-I','-B'],metadata.call_args.args[0][1:3])
                self.assertEqual(['-I','-B'],pip.call_args.args[0][1:3])
                for path in (descriptor,wheel,python,config):
                    original=path.read_bytes();path.write_bytes(original+b'changed')
                    with self.assertRaises(ValueError):verify.verify_native(proof)
                    path.write_bytes(original)
                injected=venv/'site-packages/ambient.pth';injected.parent.mkdir();injected.write_text('/ambient/packages\n')
                with self.assertRaisesRegex(ValueError,'inventory changed'):verify.verify_native(proof)
                injected.unlink()
                (venv/'lib64').unlink();(venv/'lib64').symlink_to('ambient-missing',target_is_directory=True)
                with self.assertRaisesRegex(ValueError,'inventory changed'):verify.verify_native(proof)
                (venv/'lib64').unlink();(venv/'lib64').symlink_to('lib',target_is_directory=True)
                dangling=venv/'new-link';dangling.symlink_to('missing-target')
                with self.assertRaisesRegex(ValueError,'inventory changed'):verify.verify_native(proof)
                dangling.unlink()
                extra=source/'unregistered.py';extra.write_text('')
                with self.assertRaisesRegex(ValueError,'source inventory'):verify.verify_native(proof)

    def test_readonly_retcon_verifier_rejects_source_or_api_mutation_without_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp);source=cache/'retcon/synthetic/Retcon';source.mkdir(parents=True)
            descriptor=source/'Retcon.json';descriptor.write_text(json.dumps(dict(name='Retcon',version='0.2.0',license='MIT',gameTypes=['1v1'])))
            raw=descriptor.read_bytes()
            api=cache/'retcon-api/robocode-tankroyale-bot-api-1.3.1.jar';api.parent.mkdir();api.write_bytes(b'synthetic API jar')
            lock=dict(name='Retcon',version='0.2.0',repository='synthetic',commit='synthetic',sourcePath='Retcon',license='MIT',authors=['fixture'],
                sourceSHA256=retcon.source_hash(source),files=[dict(path='Retcon.json',size=len(raw),gitBlobSHA=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())],
                botApi=dict(version='1.3.1',sha256=benchmark.digest(api)))
            lock_path=cache/'retcon-lock.json';lock_path.write_text(json.dumps(lock))
            with patch.object(verify,'CACHE',cache),patch.object(verify.retcon,'LOCK',lock_path), \
                 patch.object(verify.retcon,'prepare',side_effect=AssertionError('must never prepare')), \
                 patch.object(verify.retcon,'read_url',side_effect=AssertionError('must never download')):
                self.assertEqual('synthetic',verify.verify_retcon()['commit'])
                descriptor.write_bytes(raw+b' ')
                with self.assertRaises(ValueError):verify.verify_retcon()
                descriptor.write_bytes(raw)
                api.write_bytes(b'changed')
                with self.assertRaises(ValueError):verify.verify_retcon()


if __name__=='__main__':unittest.main()
