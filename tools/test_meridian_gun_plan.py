"""Integrity and legal-input liveness checks; no server or bot is launched."""
import copy
import gzip
import json
import tempfile
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import meridian_gun_plan as gp
import meridian_gun_plan_activate as activation


def tick(turn, events=(), heat=0, energy=100):
    return dict(type='TickEventForObserver', roundNumber=1, turnNumber=turn,
                botStates=[dict(id=1, x=100, y=100, gunHeat=heat, energy=energy)],
                events=list(events))


SCAN = dict(type='ScannedBotEvent', scannedByBotId=1, scannedBotId=2, x=500, y=100, energy=100)
FIRE = dict(type='BulletFiredEvent', bullet=dict(ownerId=1, power=1.65))


class GunPlanTests(unittest.TestCase):
    def test_schedule_matches_registered_order_and_budget(self):
        games = json.loads((gp.REG / 'games.json').read_text())
        self.assertEqual(60, sum(g['rounds'] for g in games))
        self.assertEqual(list(range(32840, 32852)), [g['port'] for g in games])
        self.assertEqual(['baseline', 'candidate'] * 3 + ['candidate', 'baseline'] * 3,
                         [g['arm'] for g in games])
        self.assertEqual([0] * 6 + [1] * 6, [g['seed'] for g in games])
        self.assertEqual('gp60-01-diamond-s0-baseline', games[0]['id'])
        self.assertEqual('gp60-12-ais-s1-baseline', games[-1]['id'])

    def test_next_tick_fire_and_terminal_decision_exclusion(self):
        result = gp.round_metrics([tick(1, [SCAN]), tick(2, [SCAN, FIRE]), tick(3, [SCAN])], 1)
        self.assertEqual(2, result['commonOpportunities'])
        self.assertEqual(1, result['noFireOpportunities'])
        self.assertEqual(1, result['longestCommonOpportunityNoFireRun'])

    def test_gaps_in_opportunities_break_run_and_reserve_is_applied(self):
        result = gp.round_metrics([tick(1, [SCAN]), tick(2), tick(3, [SCAN]),
                                   tick(4, [SCAN], energy=.1), tick(5, [SCAN])], 1)
        self.assertEqual(2, result['commonOpportunities'])
        self.assertEqual(1, result['longestCommonOpportunityNoFireRun'])
        finish = dict(SCAN, energy=.1)
        result = gp.round_metrics([tick(1, [finish], energy=1), tick(2)], 1)
        self.assertEqual(1, result['commonOpportunities'])

    def test_observer_skip_is_invalid(self):
        with self.assertRaisesRegex(ValueError, 'skipped'):
            gp.round_metrics([tick(1, [dict(type='SkippedTurnEvent', botId=1)])], 1)

    def test_receipt_cannot_be_reused_in_new_run_or_commit(self):
        head = 'a' * 40
        env = dict(PAYLOAD_SHA=head, GITHUB_RUN_ID='123', GITHUB_RUN_ATTEMPT='1')
        receipt = dict(experiment='meridian-gp60-issue90', payloadSHA=head, runID='123', runAttempt=1,
                       receiptRef=gp.RECEIPT, receiptObject='b' * 40)
        gp.validate_receipt(receipt, env, head)
        for changed in (dict(env, GITHUB_RUN_ID='124'), dict(env, GITHUB_RUN_ATTEMPT='2'),
                        dict(env, PAYLOAD_SHA='c' * 40)):
            with self.assertRaises(ValueError):
                gp.validate_receipt(receipt, changed, head)

    def test_runtime_change_is_rejected_not_normalized(self):
        baseline = json.loads((gp.ROOT / 'benchmarks/meridian-aim/profile.json').read_text())
        expected = gp.profile_expected(baseline)
        with patch.object(gp.benchmark, 'digest', side_effect=lambda path: expected['assets'][str(path.relative_to(gp.ROOT))]):
            gp.validate_profile(expected)
            changed = copy.deepcopy(expected)
            changed['image'] = 'unregistered'
            with self.assertRaises(ValueError):
                gp.validate_profile(changed)
            changed = copy.deepcopy(expected)
            changed['assets']['experiments/Meridian/src/Flight.java'] = '0' * 64
            with self.assertRaises(ValueError):
                gp.validate_profile(changed)

    def test_duplicate_durable_receipt_cannot_write_activation_or_run(self):
        head = 'a' * 40
        env = dict(PAYLOAD_SHA=head, GITHUB_RUN_ID='123', GITHUB_RUN_ATTEMPT='1', GITHUB_REPOSITORY='stephanelee/robocode')
        with patch.dict(activation.os.environ, env, clear=True), \
                patch.object(activation.comparison, 'validate_host_identity', return_value={'bootID': 'fixture-boot'}), \
                patch.object(activation.comparison.benchmark, 'digest', return_value='c' * 64), \
                patch.object(activation.subprocess, 'check_output', side_effect=[head + '\n', json.dumps(dict(sha='b'*40)),
                             subprocess.CalledProcessError(1, ['gh', 'api'])]) as command, \
                patch.object(activation.Path, 'write_text') as write:
            with self.assertRaises(subprocess.CalledProcessError):
                activation.main()
            write.assert_not_called()
            self.assertEqual(3, command.call_count)
            payload = json.loads(command.call_args.kwargs['input'])
            self.assertEqual(gp.RECEIPT, payload['ref'])
            tag = json.loads(command.call_args_list[1].kwargs['input'])
            bound = json.loads(tag['message'])
            self.assertEqual('c' * 64, bound['hostDocumentSHA256'])
            self.assertEqual({'bootID': 'fixture-boot'}, bound['hostIdentity'])

    def test_rerun_rejected_before_remote_receipt_mutation(self):
        with patch.dict(activation.os.environ, {'GITHUB_RUN_ATTEMPT': '2'}, clear=True), \
                patch.object(activation.subprocess, 'check_output') as command:
            with self.assertRaisesRegex(ValueError, 'reruns'):
                activation.main()
            command.assert_not_called()

    def test_host_document_refuses_different_boot_workspace_run_or_manifest(self):
        host = dict(bootID='boot-one', workspace='/workspace/one', runID='123', runAttempt='1',
                    runtimeProfileSHA256='a' * 64, packageContentsSHA256='b' * 64,
                    toolingManifestSHA256='c' * 64)
        with tempfile.TemporaryDirectory() as tmp, patch.object(gp, 'OUT', Path(tmp)):
            path = Path(tmp) / 'gun-plan-host.json'
            gp.save(path, host)
            receipt = dict(hostIdentity=host, hostDocumentSHA256=gp.benchmark.digest(path))
            with patch.object(gp, 'current_host_identity', return_value=host):
                self.assertEqual(host, gp.validate_host_identity(receipt))
            for field in host:
                changed = dict(host, **{field: 'different'})
                with patch.object(gp, 'current_host_identity', return_value=changed):
                    with self.assertRaisesRegex(ValueError, 'identity changed'):
                        gp.validate_host_identity(receipt)
            # Semantically identical JSON with different bytes cannot reuse the receipt.
            path.write_text(json.dumps(host))
            with patch.object(gp, 'current_host_identity', return_value=host):
                with self.assertRaisesRegex(ValueError, 'different host document'):
                    gp.validate_host_identity(receipt)

    def test_host_failure_prevents_consuming_durable_remote_receipt(self):
        head = 'a' * 40
        env = dict(PAYLOAD_SHA=head, GITHUB_RUN_ID='123', GITHUB_RUN_ATTEMPT='1')
        with patch.dict(activation.os.environ, env, clear=True), \
                patch.object(activation.subprocess, 'check_output', return_value=head + '\n') as command, \
                patch.object(activation.comparison, 'validate_host_identity', side_effect=ValueError('changed boot')):
            with self.assertRaisesRegex(ValueError, 'changed boot'):
                activation.main()
            self.assertEqual(1, command.call_count)  # Git HEAD only; no GitHub API mutation.

    def test_existing_legal_recording_reproduces_registered_starvation(self):
        path = gp.ROOT / 'local-evidence/control.battle.gz'
        if not path.exists():
            self.skipTest('Run tools/meridian_gun_plan_fixture.py to materialize fixed Git fixture')
        self.assertEqual('f0c38e81e0965b29e5c71c79ec2f2eb7a1c18232dd275172d37b1dbf6251c2c1', gp.benchmark.digest(path))
        with gzip.open(path, 'rt') as fd:
            events = [json.loads(line) for line in fd]
        ticks = [e for e in events if e['type'] == 'TickEventForObserver' and e['roundNumber'] == 5
                 and 144 <= e['turnNumber'] <= 3038]
        result = gp.round_metrics(ticks, 1)
        self.assertEqual(2894, result['commonOpportunities'])
        self.assertEqual(2894, result['noFireOpportunities'])
        self.assertEqual(2894, result['longestCommonOpportunityNoFireRun'])

    def test_complete_retained_recording_reducer_reconciles_rounds_and_totals(self):
        path = gp.ROOT / 'local-evidence/control.battle.gz'
        if not path.exists():
            self.skipTest('Fixed Git fixture not materialized')
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            (dest / 'game.battle.gz').write_bytes(path.read_bytes())
            (dest / 'markers.txt').write_text(''.join(f'{n} start\n{n} end\n' for n in range(1, 6)))
            game = dict(enemy='Diamond Port', version='1.8.28', meridianVersion='0.3.0', rounds=5)
            result = gp.metrics(dest, game)
            self.assertEqual(244, result['shots'])
            self.assertEqual(244, sum(r['shots'] for r in result['perRound']))
            self.assertEqual(result['wins'], sum(r['wins'] for r in result['perRound']))
            self.assertEqual(2894, result['perRound'][4]['longestCommonOpportunityNoFireRun'])
            self.assertAlmostEqual(result['damageDealt'], sum(r['damageDealt'] for r in result['perRound']))
            with gzip.open(dest / 'game.battle.gz', 'rt') as fd:
                events = [json.loads(line) for line in fd]
            events[0]['gameSetup']['arenaWidth'] = 801
            with gzip.open(dest / 'game.battle.gz', 'wt') as fd:
                fd.write('\n'.join(json.dumps(e) for e in events))
            with self.assertRaisesRegex(ValueError, 'setup'):
                gp.metrics(dest, game)


if __name__ == '__main__':
    unittest.main()
