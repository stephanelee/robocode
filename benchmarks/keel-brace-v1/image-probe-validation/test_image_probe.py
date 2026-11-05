"""Pure fake-file/fake-proc tests. Never starts a process, network, JVM, bot or engine."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('keel_brace_image_probe_under_test', ROOT / 'tools/keel_brace_image_probe.py')
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)
GEN_SPEC = importlib.util.spec_from_file_location('probe_workflow_generator', Path(__file__).with_name('make_workflow.py'))
GEN = importlib.util.module_from_spec(GEN_SPEC)
GEN_SPEC.loader.exec_module(GEN)
DETAIL = 'Image: ubuntu-26.04\nVersion: 20261025.179.1\nIncluded Software: https://example.invalid/software\nImage Release: https://example.invalid/release'


def image(groups=None):
    return json.dumps(groups if groups is not None else [{'group': 'Operating System', 'detail': 'Ubuntu'}, {'group': 'Runner Image', 'detail': DETAIL}]).encode()


class ImageProbePureTests(unittest.TestCase):
    def test_source_backed_array_and_full_detail(self):
        parsed = PROBE.parse_image_json(image())
        self.assertEqual([], parsed['errors'])
        self.assertEqual(DETAIL, parsed['runnerImageRecords'][0]['detail'])
        self.assertEqual('20261025.179.1', parsed['claim']['version'])
        self.assertEqual('ubuntu-26.04', parsed['claim']['image'])

    def test_setup_info_case_binding_and_explicit_utf8_bom(self):
        parsed = PROBE.parse_image_json(b'\xef\xbb\xbf' + image([{'Group': 'Runner Image', 'Detail': DETAIL}]))
        self.assertEqual([], parsed['errors'])
        self.assertEqual('UTF-8-BOM', parsed['encoding'])
        self.assertEqual('20261025.179.1', parsed['claim']['version'])

    def test_duplicate_json_keys_never_select_a_claim(self):
        parsed = PROBE.parse_image_json(b'[{"group":"Runner Image","group":"Runner Image","detail":"x"}]')
        self.assertIsNone(parsed['claim'])
        self.assertIn('duplicate JSON key', parsed['errors'][0]['message'])

    def test_case_alias_ambiguity_never_selects_a_claim(self):
        parsed = PROBE.parse_image_json(image([{'group': 'Runner Image', 'Group': 'Runner Image', 'detail': DETAIL}]))
        self.assertIsNone(parsed['claim'])
        self.assertTrue(parsed['errors'])

    def test_two_runner_image_groups_or_duplicate_version_are_ambiguous(self):
        for data in [image([{'group': 'Runner Image', 'detail': DETAIL}] * 2), image([{'group': 'Runner Image', 'detail': DETAIL + '\nVersion: 20261025.179.1'}])]:
            self.assertIsNone(PROBE.parse_image_json(data)['claim'])

    def test_wrong_shapes_nonfinite_bad_utf8_and_missing_fields_are_reported(self):
        for data in [b'{"ImageVersion":"x"}', b'[NaN]', b'\xff', b'[{"group":"Runner Image"}]', image([{'group': 'Runner Image', 'detail': 'Version: x'}])]:
            parsed = PROBE.parse_image_json(data)
            self.assertIsNone(parsed['claim'])
            self.assertTrue(parsed['errors'])

    def test_group_and_detail_limits_keep_diagnostic_state(self):
        parsed = PROBE.parse_image_json(image([{'group': 'x', 'detail': 'x'}] * (PROBE.JSON_GROUP_LIMIT + 1)))
        self.assertIsNone(parsed['claim'])
        self.assertIsNone(parsed['runnerImageCount'])
        parsed = PROBE.parse_image_json(image([{'group': 'Runner Image', 'detail': 'x' * (PROBE.DETAIL_CHAR_CAP + 1)}]))
        self.assertIsNone(parsed['claim'])
        self.assertEqual(PROBE.DETAIL_CHAR_CAP, len(parsed['runnerImageRecords'][0]['detailPrefix']))

    def test_environment_values_are_literal_and_sources_remain_separate(self):
        parsed = PROBE.parse_environment_file(b'PATH=$OTHER\nImageVersion="20261026.180.1"\nImageOS=ubuntu26\n')
        self.assertEqual([], parsed['errors'])
        self.assertEqual('20261026.180.1', parsed['claims']['ImageVersion']['value'])
        self.assertEqual('20261025.179.1', PROBE.parse_image_json(image())['claim']['version'])
        process = PROBE.process_environment_claims({'ImageOS': '', 'SECRET': 'never-output'})
        self.assertEqual({'present': True, 'value': ''}, process['ImageOS'])
        self.assertEqual({'present': False, 'value': None}, process['ImageVersion'])
        self.assertNotIn('SECRET', json.dumps(process))

    def test_environment_duplicates_malformed_and_expansion_have_no_fallback(self):
        cases = [b'ImageVersion=1\nImageVersion=1\nImageOS=ubuntu26\n', b'export ImageVersion=1\nImageOS=ubuntu26\n', b'ImageVersion=$SECRET\nImageOS=ubuntu26\n', b'ImageVersion="$(danger)"\nImageOS=ubuntu26\n', b'ImageVersion="unclosed\nImageOS=ubuntu26\n']
        for data in cases:
            parsed = PROBE.parse_environment_file(data)
            self.assertIsNone(parsed['claims']['ImageVersion']['value'])
            self.assertTrue(parsed['errors'])
        empty = PROBE.parse_environment_file(b'ImageVersion=\nImageOS=ubuntu26\n')
        self.assertEqual('', empty['claims']['ImageVersion']['value'])

    def test_raw_bytes_stat_and_hash_are_retained_before_any_parse(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td).resolve(); source = base / 'image.json'; source.write_bytes(b'not JSON\n')
            artifact = PROBE.Artifact(base / 'out')
            row, data = PROBE.capture_file(source, 'image', artifact, {str(source)})
            self.assertEqual(b'not JSON\n', data)
            self.assertTrue(row['bytesComplete'])
            self.assertEqual(data, (artifact.root / row['raw']['artifactPath']).read_bytes())
            self.assertEqual(PROBE.sha(data), row['raw']['sha256'])
            self.assertEqual(source.stat().st_uid, row['statBefore']['uid'])
            self.assertEqual(source.stat().st_mtime_ns, row['statBefore']['mtimeNs'])
            self.assertTrue(PROBE.parse_image_json(data)['errors'])

    def test_only_allowed_resolved_metadata_paths_are_read(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td).resolve(); target = base / 'not-allowed'; target.write_bytes(b'private')
            link = base / 'environment'; link.symlink_to(target)
            artifact = PROBE.Artifact(base / 'out')
            row, data = PROBE.capture_file(link, 'environment', artifact, {str(link)})
            self.assertIsNone(data)
            self.assertEqual('resolvedPathOutsideMetadataAllowlist', row['errors'][0]['code'])
            self.assertEqual(0, artifact.bytes)
            allowed, data = PROBE.capture_file(link, 'allowed-alias', artifact, {str(target)})
            self.assertEqual(b'private', data)
            self.assertEqual(str(target), allowed['resolvedPath'])

    def test_missing_loop_and_nonregular_path_errors_are_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td).resolve(); artifact = PROBE.Artifact(base / 'out')
            row, data = PROBE.capture_file(base / 'missing', 'missing', artifact, set())
            self.assertFalse(row['exists']); self.assertTrue(row['errors']); self.assertIsNone(data)
            loop = base / 'loop'; loop.symlink_to(loop)
            row, data = PROBE.capture_file(loop, 'loop', artifact, {str(loop)})
            self.assertTrue(row['errors']); self.assertIsNone(data)
            row, data = PROBE.capture_file(base, 'directory', artifact, {str(base)})
            self.assertEqual('notRegularMetadataFile', row['errors'][0]['code'])

    def test_oversized_metadata_retains_prefix_not_a_whole_hash_claim(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td).resolve(); source = base / 'oversize'; source.write_bytes(b'x' * (PROBE.FILE_CAP + 5))
            artifact = PROBE.Artifact(base / 'out')
            row, data = PROBE.capture_file(source, 'oversize', artifact, {str(source)})
            self.assertIsNone(data); self.assertFalse(row['bytesComplete'])
            self.assertFalse(row['raw']['sourceBytesComplete'])
            self.assertEqual('captured-prefix-only', row['rawSHA256Scope'])
            self.assertEqual(PROBE.FILE_CAP, row['raw']['bytes'])
            self.assertTrue(row['errors'])

    def test_two_phase_payload_and_zip_overhead_budget_are_bounded(self):
        with tempfile.TemporaryDirectory() as td:
            total = 0
            for phase in ['before-tools', 'after-tools']:
                artifact = PROBE.Artifact(Path(td) / phase)
                for index in range(6): artifact.write('raw/' + str(index), b'x' * PROBE.FILE_CAP)
                self.assertLessEqual(artifact.bytes, PROBE.RAW_PHASE_CAP)
                report = artifact.json('report.json', {'text': 'x' * (PROBE.FILE_CAP + 1)}, final=True)
                self.assertFalse(report['complete'])
                artifact.status({'diagnostic': 'report prefix retained, no acceptance'})
                self.assertLessEqual(artifact.bytes, PROBE.PHASE_CAP)
                total += artifact.bytes
            self.assertLessEqual(total + PROBE.ZIP_OVERHEAD_RESERVE, PROBE.ARTIFACT_CAP)

    def fake_process(self, base, pid, comm, executable=None):
        directory = base / str(pid); directory.mkdir()
        (directory / 'comm').write_bytes(comm.encode() + b'\n')
        if executable is not None: (directory / 'exe').symlink_to(executable)
        return directory

    def test_runner_roots_derive_only_from_named_observed_executables(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td).resolve(); proc = base / 'proc'; proc.mkdir()
            runner = base / 'observed-install'; (runner / 'bin').mkdir(parents=True)
            for name in ['Runner.Worker', 'Runner.Listener']: (runner / 'bin' / name).write_bytes(b'not executed')
            self.fake_process(proc, 10, 'unrelated')  # exe absent: must never be inspected
            self.fake_process(proc, 11, 'Runner.Worker', runner / 'bin/Runner.Worker')
            self.fake_process(proc, 12, 'Runner.Listener', runner / 'bin/Runner.Listener')
            d = PROBE.discover_runner_roots(proc)
            self.assertEqual([], d['errors']); self.assertEqual([str(runner)], d['roots'])
            self.assertEqual(2, len(d['matches'])); self.assertEqual(1, d['nonmatchingCount'])
            self.assertEqual(str(runner / '.setup_info'), d['matches'][0]['setupInfoPath'])
            self.assertEqual(PROBE.sha(b'Runner.Worker\n'), d['matches'][0]['commSHA256'])
            self.assertTrue(all(not m['errors'] for m in d['matches']))

    def test_no_root_guessed_when_executable_layout_is_not_observed(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td).resolve(); dotnet = base / 'dotnet'; dotnet.write_bytes(b'not executed')
            self.fake_process(base, 1, 'Runner.Worker', dotnet)
            d = PROBE.discover_runner_roots(base)
            self.assertEqual([], d['roots']); self.assertIsNone(d['matches'][0]['setupInfoPath'])
            self.assertTrue(d['matches'][0]['errors']); self.assertTrue(d['errors'])

    def test_discovery_failure_cap_and_deadline_are_not_global_absence_claims(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td).resolve(); (base / '1').mkdir()  # comm read failure preserved
            d = PROBE.discover_runner_roots(base)
            self.assertTrue(d['errors']); self.assertEqual([], d['roots'])
            with mock.patch.object(PROBE, 'PROC_LIMIT', 0):
                d = PROBE.discover_runner_roots(base)
            self.assertTrue(d['truncated'])
            d = PROBE.discover_runner_roots(base, deadline=time.monotonic() - 1)
            self.assertTrue(d['truncated']); self.assertEqual(0, d['scannedEntries'])

    def test_existing_output_cannot_be_mixed_or_paths_escape(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td).resolve(); artifact = PROBE.Artifact(base / 'out')
            artifact.write('x', b'bytes')
            with self.assertRaises(ValueError): PROBE.Artifact(base / 'out')
            with self.assertRaises(ValueError): artifact.write('../escape', b'bytes')

    def test_workflow_inline_source_is_byte_identical_and_order_is_frozen(self):
        source_bytes = (ROOT / 'tools/keel_brace_image_probe.py').read_bytes()
        self.assertNotIn(b'\r', source_bytes); self.assertNotIn(b'${{', source_bytes)
        source = source_bytes.decode('utf-8')
        workflow = (ROOT / '.github/workflows/keel-brace-image-probe.yml').read_text()
        self.assertEqual(GEN.workflow(source), workflow)
        marker = GEN.MARKER
        body = workflow.split("<<'" + marker + "'\n", 1)[1].split('          ' + marker + '\n', 1)[0]
        self.assertEqual(source, ''.join(line[10:] if line.startswith('          ') else line for line in body.splitlines(keepends=True)))
        self.assertNotIn('actions/checkout', workflow)
        before = workflow.index('--phase before-tools')
        java = workflow.index('uses: actions/setup-java@v4')
        python = workflow.index('uses: actions/setup-python@v5')
        after = workflow.index('--phase after-tools')
        upload = workflow.index('uses: actions/upload-artifact@v4')
        self.assertLess(before, java); self.assertLess(java, python); self.assertLess(python, after); self.assertLess(after, upload)
        self.assertIn('timeout-minutes: 10', workflow); self.assertIn('runs-on: ubuntu-26.04', workflow)
        self.assertIn("github.run_attempt == 1", workflow)
        self.assertIn("KEEL-BRACE-IMAGE-PROBE-CLEARED {0}", workflow)
        self.assertNotIn('gradlew', workflow); self.assertNotIn('java -', workflow)

    def test_complete_mocked_phase_keeps_conflicts_and_never_grants_runtime_acceptance(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td).resolve(); sources = base / 'sources'; sources.mkdir()
            files = {}
            payloads = {'imagedata': image(), 'etc_environment': b'ImageOS=ubuntu26\nImageVersion=20261026.180.1\n',
                        'etc_os_release': b'ID=ubuntu\nVERSION_ID="26.04"\n', 'usr_lib_os_release': b'ID=ubuntu\nVERSION_ID="26.04"\n',
                        'etc_lsb_release': b'DISTRIB_ID=Ubuntu\n', 'boot_id': b'11111111-2222-3333-4444-555555555555\n'}
            for name, data in payloads.items():
                p = sources / name; p.write_bytes(data); files[name] = str(p)
            runner = base / 'observed-runner'; runner.mkdir(); (runner / '.setup_info').write_bytes(image())
            discovery = {'roots': [str(runner)], 'matches': [], 'errors': [], 'truncated': False}
            env = {'ImageOS': 'ubuntu26', 'ImageVersion': '20261026.180.1', 'GITHUB_TOKEN': 'never-retain',
                   'KEEL_BRACE_IMAGE_PROBE_HEAD_SHA': 'test-fixture-head'}
            with mock.patch.object(PROBE, 'FIXED_FILES', files), mock.patch.object(PROBE, 'discover_runner_roots', return_value=discovery):
                result = PROBE.collect(base / 'capture/before-tools', 'before-tools', env)
            self.assertEqual(0, result, 'successful observation is deliberately not image agreement/admission')
            report = json.loads((base / 'capture/before-tools/report.json').read_text())
            self.assertEqual('before-tools', report['phase']); self.assertTrue(report['noRuntimeAcceptance'])
            self.assertEqual(0, report['competitiveGames']); self.assertTrue(report['notImmutableAttestation'])
            self.assertEqual('20261026.180.1', report['processImageEnvironment']['ImageVersion']['value'])
            self.assertEqual('20261025.179.1', report['captures']['imagedata']['parsedClaims']['version'])
            self.assertEqual('20261025.179.1', report['captures']['setup_info_0']['parsedClaims']['version'])
            self.assertEqual('20261026.180.1', report['captures']['etc_environment']['parsedClaims']['ImageVersion']['value'])
            self.assertEqual(payloads['boot_id'].decode(), report['captures']['boot_id']['literalASCII'])
            self.assertNotIn('never-retain', json.dumps(report)); self.assertNotIn('GITHUB_TOKEN', json.dumps(report))
            self.assertEqual((ROOT / 'tools/keel_brace_image_probe.py').read_bytes(), (base / 'capture/before-tools/collector.py').read_bytes())

    def test_collect_phase_contract_rejects_bad_scope_without_reading_sources(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError): PROBE.collect(Path(td) / 'bad', 'bad', {})
            with self.assertRaises(ValueError): PROBE.collect(Path(td) / 'wrong-directory', 'before-tools', {})


if __name__ == '__main__':
    unittest.main()
