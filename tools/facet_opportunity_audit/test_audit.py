"""Synthetic audit semantics, not acquired firing opportunities or strength tests."""
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import audit

PLAN = ('sequence=7 fire=false plan=Plan[angle=0.0, power=2.0, target=2, created=1, expires=3, '
        'tolerance=0.008726646259971648, origin=V[x=400.0, y=300.0], observationTurn=1, '
        'targetPosition=V[x=600.0, y=300.0], version=0, '
        'targetBody=Body[p=V[x=600.0, y=300.0], heading=0.0, speed=0.0, gun=0.0, radar=0.0, heat=0.0], '
        'policy=Policy[population=1, target=2, range=200.0, targetEnergy=100.0, observed=1], '
        'evidence=gun=recent mode=duel, cohort=null]')


class Parser(unittest.TestCase):
    def test_full_record_preserves_units_and_nested_identity(self):
        p = audit.parse_plan(PLAN)
        self.assertEqual(p['origin'], (400, 300))
        self.assertEqual(p['targetBody']['position'], (600, 300))
        self.assertEqual(p['gun'], 'recent')
        self.assertEqual(p['created'], 1)
        self.assertEqual(p['expires'], 3)
        self.assertAlmostEqual(p['tolerance'], math.radians(.5))

    def test_null_and_incomplete_record_are_distinct(self):
        self.assertIsNone(audit.parse_plan('fire=false plan=null'))
        with self.assertRaises(ValueError): audit.parse_plan('plan=1:2:3 angle=0')

    def test_field_boundary_does_not_confuse_turn_and_expected_turn(self):
        self.assertEqual(audit.numeric('expectedOriginTurn=99 turn=2', 'turn'), 2)
        with self.assertRaises(ValueError): audit.numeric('expectedOriginTurn=99', 'turn')
        with self.assertRaises(ValueError): audit.numeric('angle=NaN', 'angle')

    def test_archive_allowlist_cannot_open_logs_or_models(self):
        evidence = audit.Evidence.__new__(audit.Evidence)
        evidence.allowed = {'registration.json'}
        for name in ('runner.log', 'diamond-port-g01/FACET_model.dat', '../registration.json'):
            with self.assertRaises(ValueError): evidence.open(name)


class ActualJarGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='facet-audit-test-')
        jar = Path(os.environ.get('FACET_AUDIT_JAR',
                                  Path(__file__).resolve().parents[2] / 'experiments/Facet/Facet.jar'))
        cls.replay = audit.Replay(Path(os.environ['FACET_AUDIT_JAVA_HOME']), jar, Path(cls.directory.name))

    @classmethod
    def tearDownClass(cls):
        cls.replay.close(); cls.directory.cleanup()

    def fixture(self):
        return (dict(turn=2, x=400, y=300, gun=0, heat=0, energy=100, population=1),
                audit.parse_plan(PLAN),
                dict(turn=2, x=600, y=300, heading=0, speed=0, energy=100, alive=True))

    def test_valid_plan_and_exact_zero_heading(self):
        result = self.replay.check(*self.fixture(), False)
        self.assertTrue(result['valid']); self.assertEqual(result['failed'], [])

    def test_overlapping_predicates_do_not_invent_first_rejection(self):
        own, plan, target = self.fixture()
        own.update(heat=.1, gun=.1, energy=1)
        result = self.replay.check(own, plan, target, True)
        self.assertEqual(set(result['failed']), {'pending', 'heat', 'alignment', 'energy', 'powerPolicy'})
        self.assertFalse(result['valid'])

    def test_expiry_scan_and_displacement_are_independent(self):
        own, plan, target = self.fixture()
        own.update(turn=5, x=380); target['turn'] = 1
        result = self.replay.check(own, plan, target, False)
        self.assertEqual(set(result['failed']), {'planAge', 'scanAge', 'ownDisplacement'})

    def test_legal_stop_can_invalidate_long_horizon_analytic_drift(self):
        own, plan, target = self.fixture()
        plan['targetBody'].update(heading=math.pi/2, speed=2)
        target.update(heading=math.pi/2, speed=0)
        result = self.replay.check(own, plan, target, False)
        self.assertEqual(result['failed'], ['analyticDrift'])
        self.assertGreater(result['drift'], result['half'])

    def test_trace_join_uses_prior_plan_and_clears_accepted_request(self):
        rows = []
        def event(turn, kind, value):
            rows.append(dict(round=1, turn=turn, kind=kind, value=value))
        def plan(turn, angle):
            return PLAN.replace('created=1', f'created={turn}').replace('expires=3', f'expires={turn+2}').replace('angle=0.0', f'angle={angle}')
        for turn, heat, angle, fire in ((1, 0, 0, False), (2, 0, .1, False),
                                        (3, 0, .1, True), (4, 1.3, .1, False)):
            event(turn, 'own.state', f'x=400 y=300 heading=0 speed=0 gun=.1 heat={heat} energy=100 population=1')
            if turn == 4:
                event(turn, 'BulletFiredEvent', '42 callbackSequence=9 ownStateTurn=4')
                event(turn, 'fire.confirmed', 'id=42 request=3 heading=5.729577951308233 position=400,300 gun=recent')
            event(turn, 'scan.observed', 'id=2 x=600 y=300 directionDeg=0 speed=0 energy=100')
            event(turn, 'command.prepared', plan(turn, angle).replace('fire=false', f'fire={str(fire).lower()}'))
            if fire:
                event(turn, 'fire.submitted', 'accepted=true')
            event(turn, 'command.submitted', f'own=fixture fire={str(fire).lower()} firepower={2 if fire else 0}')
        class Fixture:
            def open(self, name):
                return io.BytesIO(('\n'.join(map(json.dumps, rows))+'\n').encode())
            def rows(self, name, cap):
                return iter((i, dict(round=i, shotsConfirmed=1, counters={})) for i in range(1, 21))
        with patch.object(audit, 'MAX_LINES', len(rows)):
            result = audit.analyze_game(Fixture(), 1, self.replay)
        self.assertEqual(result['counts']['replayMatchesIntent'], 2)
        self.assertEqual(result['counts']['noPriorActivePlan'], 2)
        self.assertEqual(result['counts']['nativeHeadingOriginNextTickPass'], 1)
        self.assertEqual(result['counts']['submittedFireCommands'], 1)
        self.assertEqual(result['observedOutcomes'], {'notObservedBeforeCapOrTerminal': 1})
        last_command = rows.pop()
        self.assertEqual(last_command['kind'], 'command.submitted')
        with patch.object(audit, 'MAX_LINES', len(rows)):
            cut = audit.analyze_game(Fixture(), 1, self.replay)
        self.assertEqual(cut['counts']['unsubmittedPreparedAtCap'], 1)
        self.assertFalse(cut['unsubmittedPreparedAtCapEvidence']['fireIntent'])
        rows.append(last_command)
        event(4, 'BulletHitBulletEvent', '42:9:99 callbackSequence=11')
        event(4, 'BulletHitBotEvent', '1:42 callbackSequence=12')
        with patch.object(audit, 'MAX_LINES', len(rows)):
            mixed = audit.analyze_game(Fixture(), 1, self.replay)
        self.assertEqual(mixed['observedOutcomes'], {'BulletHitBotEvent+BulletHitBulletEvent': 1})
        self.assertEqual(mixed['counts']['multipleOutcomeEvidenceRecords'], 1)


if __name__ == '__main__': unittest.main()
