"""Synthetic integrity guards and retrospective unchanged-recording reducer parity; zero games."""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import benchmark
import meridian_peak as peak
import meridian_peak_activate as activate
import meridian_peak_common as common
import meridian_peak_profile as profile
import meridian_peak_state as state


def synthetic_snapshot(number=0):
    return dict(schemaVersion=1, name='MeridianPeak', version='0.1.0', seed=61000, checkpointLoaded=False,
        round=number, completedRounds=number, revision=0 if number == 0 else 2, sealed=number > 0,
        coldStateSHA256=None, cumulativeCrossings=0, cumulativeAcceptedHits=0,
        stats={**{key: 0 for key in state.STATS_INTS}, **{key: 0. for key in state.STATS_FLOATS}},
        models=dict(hitAngles=dict(weights=[0.] * 21, effectiveTotal=0.),
            crossingGun=dict(capacity=192, count=0, next=0, examples=[], inFlight=[], requested=None), gunSkill=[.2, .3, .25], pendingTrials=[]))


def synthetic_evidence(root, *, shots=1, crossings=0, hits_taken=0):
    root.mkdir()
    cold = root / 'cold.json'
    cold.write_text(json.dumps(synthetic_snapshot()) + '\n')
    value = synthetic_snapshot(1)
    value['coldStateSHA256'] = benchmark.digest(cold)
    value['stats'].update(confirmedShots=shots, shotEnergy=shots*.1, hitsTaken=hits_taken, damageTaken=hits_taken*.4,
                          crossingLabels=crossings, acceptedHitLabels=hits_taken)
    value.update(cumulativeCrossings=crossings, cumulativeAcceptedHits=hits_taken)
    mass = (1-.95**hits_taken)/.05
    value['models']['hitAngles']['effectiveTotal'] = mass
    value['models']['hitAngles']['weights'][10] = mass
    bank = value['models']['crossingGun']
    bank.update(count=min(192, crossings), next=crossings % 192,
                examples=[dict(index=i, context=dict(lateral=0., radial=0., wall=1., flight=20.), offset=0.) for i in range(min(192, crossings))])
    for name in ('round-001.json', 'endpoint.json'):
        (root / name).write_text(json.dumps(value) + '\n')
    labels=[]
    for i in range(crossings):
        launch_turn=30+i*10
        labels.append(dict(round=1,type='crossing',label=dict(turn=launch_turn+20,
            launch=dict(turn=launch_turn,x=100.,y=100.,power=.1,bearing=0.,sign=1.,range=423.55,
                        context=dict(lateral=0.,radial=0.,wall=1.,flight=20.)),
            enemy=dict(x=523.55,y=100.,h=0.,v=0.),previousTurn=launch_turn+19,
            previous=dict(x=523.55,y=100.,h=0.,v=0.),reason='RESOLVED',offset=0.)))
    (root/'labels.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in labels))
    official = dict(perRound=[dict(turns=3000, shots=shots, hits=0, hitsTaken=hits_taken, skips=0,
                                  shotEnergy=shots*.1, damageDealt=0., damageTaken=hits_taken*.4,
                                  callbackDeliverableOutgoingHits=0,callbackDeliverableOutgoingDamage=0.,
                                  observerOnlyOutgoingHits=0,observerOnlyOutgoingDamage=0.)])
    markers = ['1 start', *[f'1 fire {30+i*10} 0.1' for i in range(shots)],
               *['1 hit ACCEPTED']*hits_taken, *([f'1 crossings {crossings}'] if crossings else []), '1 end']
    return value, official, markers


def scored_row(share=.6, wins=3, incoming=20.):
    return dict(status='valid', skips=0, shots=10, enemyShots=20, share=share, wins=wins, damagePer100EnemyShots=incoming)


class PeakIntegrityTests(unittest.TestCase):
    def test_exact_registered_schedules_and_fixed_stage_receipts(self):
        rows = json.loads((common.REG / 'games.json').read_text())
        self.assertEqual(25, sum(g['rounds'] for g in rows if g['stage'] == 'screen'))
        self.assertEqual(180, sum(g['rounds'] for g in rows if g['stage'] == 'confirmation'))
        self.assertEqual(list(range(32900, 32905)) + list(range(32910, 32919)), [g['port'] for g in rows])
        self.assertEqual([61000]*5 + [61011, 61012, 61013, 61021, 61022, 61023, 61031, 61032, 61033], [g['seed'] for g in rows])
        self.assertEqual('Meridian', rows[0]['ownName'])
        self.assertEqual('0.3.0', rows[0]['ownVersion'])
        self.assertTrue(all(g['ownName']=='MeridianPeak' and g['ownVersion']=='0.1.0' for g in rows[1:]))
        self.assertIsNone(rows[4]['version'], 'Walls version must come from verified ZIP, not an invented literal')
        self.assertNotEqual(peak.receipt_ref('screen'), peak.receipt_ref('confirmation'))

    def test_screen_thresholds_are_strict_and_every_guard_required(self):
        rows = [scored_row(.5, 0, 40), *[scored_row() for _ in range(4)]]
        self.assertTrue(common.passes_screen(rows))
        for index, key, value in ((1,'share',.5),(1,'damagePer100EnemyShots',40),(2,'share',.35),(3,'wins',0),(4,'shots',0),(0,'enemyShots',0),(4,'status','invalid')):
            changed=copy.deepcopy(rows);changed[index][key]=value
            self.assertFalse(common.passes_screen(changed), (index,key,value))

    def test_confirmation_requires_each_opponent_majority_and_no_screen_pooling(self):
        rows = [dict(scored_row(.6, 11), enemy=enemy) for enemy in ('Diamond Port','Retcon','AISurvivor') for _ in range(3)]
        self.assertTrue(common.confirmation_decision(rows)['passAll'])
        changed=copy.deepcopy(rows);changed[6]['wins']=8
        self.assertFalse(common.confirmation_decision(changed)['passAll'])  # exactly30 is insufficient
        self.assertFalse(common.confirmation_decision(rows+rows[:1])['passAll'])

    def test_latest_sealed_revision_and_capped_bank_distinct_from_cumulative_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'state'
            value,official,markers=synthetic_evidence(root,shots=193,crossings=193,hits_taken=21)
            result=state.validate(root,dict(seed=61000,rounds=1),markers,official)
            self.assertEqual(193,result['cumulativeCrossings'])
            self.assertEqual(2,result['sealedRounds'][0]['revision'])
            self.assertEqual(192,value['models']['crossingGun']['count'])
            self.assertLess(value['models']['hitAngles']['effectiveTotal'],21)
            value['revision']=8
            for name in ('round-001.json','endpoint.json'):(root/name).write_text(json.dumps(value)+'\n')
            self.assertEqual(8,state.validate(root,dict(seed=61000,rounds=1),markers,official)['sealedRounds'][0]['revision'])

    def test_double_round_advance_missing_counter_nonfinite_and_unsealed_endpoint_rejected(self):
        for mutation in ('double','missing','nan','endpoint'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'state';value,official,markers=synthetic_evidence(root)
                if mutation=='double':value['completedRounds']=2
                if mutation=='missing':del value['stats']['kernelEvaluations']
                if mutation=='nan':value['models']['hitAngles']['weights'][0]=float('nan')
                if mutation=='endpoint':value['sealed']=False
                (root/'endpoint.json').write_text(json.dumps(value)+'\n')
                if mutation!='endpoint':(root/'round-001.json').write_text(json.dumps(value)+'\n')
                with self.assertRaises(ValueError):state.validate(root,dict(seed=61000,rounds=1),markers,official)

    def test_missing_callback_or_duplicate_fire_confirmation_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'state';value,official,markers=synthetic_evidence(root)
            with self.assertRaises(ValueError):state.validate(root,dict(seed=61000,rounds=1),markers+['1 fire 30 0.1'],official)
            official['perRound'][0]['shots']=2
            with self.assertRaises(ValueError):state.validate(root,dict(seed=61000,rounds=1),markers,official)

    def test_masked_label_journal_is_required_and_unknown_reasons_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'state';value,official,markers=synthetic_evidence(root)
            label=dict(round=1,type='trial',label=dict(observedTurn=42,reason='MISSING_SCAN',
                trial=dict(due=40,angles=[0.,10.,20.],x=100.,y=100.),enemy=dict(x=500.,y=100.,h=0.,v=0.)))
            (root/'labels.jsonl').write_text(json.dumps(label)+'\n')
            with self.assertRaisesRegex(ValueError,'journal counts'):state.validate(root,dict(seed=61000,rounds=1),markers,official)
            value['stats']['trialMissingScan']=1
            for name in ('round-001.json','endpoint.json'):(root/name).write_text(json.dumps(value)+'\n')
            self.assertEqual(1,state.validate(root,dict(seed=61000,rounds=1),markers,official)['labelRows'])
            label['label']['reason']='UNDECLARED'
            (root/'labels.jsonl').write_text(json.dumps(label)+'\n')
            with self.assertRaises(ValueError):state.validate(root,dict(seed=61000,rounds=1),markers,official)

    def test_boolean_is_not_a_numeric_model_weight(self):
        value=synthetic_snapshot();value['models']['hitAngles']['weights'][0]=False
        with self.assertRaisesRegex(ValueError,'numeric'):state.snapshot(value,dict(seed=61000),0,None)

    def test_same_tick_death_delivery_and_posthumous_observer_only_hits(self):
        def hit():return dict(type='BulletHitBotEvent',bullet=dict(ownerId=1,power=.1),victimId=2,damage=.4)
        ticks=[dict(botStates=[dict(id=1,energy=-1),dict(id=2)],events=[hit()]),
               dict(botStates=[dict(id=2)],events=[hit()])]
        result=common.counters(ticks,1)
        self.assertEqual(2,result['hits'])
        self.assertEqual(.8,result['damageDealt'])
        self.assertEqual(1,result['callbackDeliverableOutgoingHits'])
        self.assertEqual(.4,result['callbackDeliverableOutgoingDamage'])
        self.assertEqual(1,result['observerOnlyOutgoingHits'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'state';value,official,markers=synthetic_evidence(root)
            value['stats'].update(hits=1,damageDealt=.4)
            official['perRound'][0].update(hits=2,damageDealt=.8,callbackDeliverableOutgoingHits=1,
                callbackDeliverableOutgoingDamage=.4,observerOnlyOutgoingHits=1,observerOnlyOutgoingDamage=.4)
            for name in ('round-001.json','endpoint.json'):(root/name).write_text(json.dumps(value)+'\n')
            state.validate(root,dict(seed=61000,rounds=1),markers,official)
            value['stats'].update(hits=0,damageDealt=0.)  # A same-tick-death callback cannot be excused as posthumous.
            for name in ('round-001.json','endpoint.json'):(root/name).write_text(json.dumps(value)+'\n')
            with self.assertRaises(ValueError):state.validate(root,dict(seed=61000,rounds=1),markers,official)

    def test_unregistered_runtime_rejected_without_asset_normalization(self):
        expected=profile.allowed_runtime()
        self.assertEqual(2,len(expected))
        changed=copy.deepcopy(expected[0]);changed['image']='unregistered'
        self.assertNotIn(changed,expected)
        self.assertEqual(profile.PACKAGE_SHA,benchmark.digest(common.ROOT/'benchmarks/meridian-gun-plan/package-contents.json'))

    def test_payload_rejects_protocol_family_or_package_metadata_tampering(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(profile,'REG',Path(tmp)), \
             patch.object(profile,'participant_assets',return_value={}),patch.object(profile,'closure',return_value={}):
            reg=Path(tmp);(reg/'games.json').write_text('[]')
            family=profile.allowed_runtime()
            frozen=dict(schemaVersion=1,controlCommit=profile.BASE,controlJarSHA256=profile.CONTROL_JAR,
                protocolSHA256=benchmark.digest(common.ROOT/'docs/improvements/meridian-peak-v1.md'),
                allowedRuntime=family,packageSHA256=profile.PACKAGE_SHA,assets={},tooling={},walls={},
                gamesSHA256=benchmark.digest(reg/'games.json'))
            actual=dict(runtime=family[0],assets={},walls={})
            common.save(reg/'freeze.json',frozen)
            profile.validate_payload(actual)
            for field,value in (('protocolSHA256','0'*64),('allowedRuntime',[]),('packageSHA256','1'*64),('controlCommit','2'*40)):
                common.save(reg/'freeze.json',dict(frozen,**{field:value}))
                with self.assertRaises(ValueError):profile.validate_payload(actual)

    def test_walls_identity_and_files_come_from_verified_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'synthetic.zip'
            with zipfile.ZipFile(path,'w') as z:
                z.writestr('samples/Walls/Walls.json',json.dumps(dict(name='Walls',version='synthetic-version')))
                z.writestr('samples/Walls/Walls.java','// synthetic fixture only')
                z.writestr('samples/lib/api.jar','synthetic fixture only')
            with self.assertRaises(ValueError):profile.walls_manifest(path)
            with patch.object(profile.benchmark,'digest',return_value=benchmark.ASSETS['sample-bots-Java-1.4.0.zip']):
                result=profile.walls_manifest(path)
            self.assertEqual('synthetic-version',result['version'])
            self.assertEqual(3,len(result['files']))

    def test_atomic_duplicate_stage_receipt_never_writes_activation(self):
        head='a'*40;host={'stage':'screen','bootID':'one'}
        with patch.dict(os.environ,dict(GITHUB_RUN_ATTEMPT='1',GITHUB_RUN_ID='123',GITHUB_REPOSITORY='stephanelee/robocode',PAYLOAD_SHA=head),clear=True), \
             patch.object(activate.subprocess,'check_output',return_value=head+'\n'), \
             patch.object(activate,'validate_payload'),patch.object(activate,'check_host',return_value=host), \
             patch.object(activate.PROVENANCE.__class__,'read_text',return_value='{}'), \
             patch.object(activate.benchmark,'digest',return_value='b'*64), \
             patch.object(activate,'api',side_effect=[{'sha':'c'*40},subprocess.CalledProcessError(1,['gh'])]),patch.object(activate,'save') as save:
            with self.assertRaises(subprocess.CalledProcessError):activate.activate('screen')
            save.assert_not_called()

    def test_stage_receipt_cannot_cross_run_stage_or_payload(self):
        head='a'*40
        with tempfile.TemporaryDirectory() as tmp,patch.object(peak,'REG',Path(tmp)),patch.object(peak,'check_host'):
            (Path(tmp)/'freeze.json').write_text('{}')
            receipt=dict(experiment='meridian-peak-v1',stage='screen',payloadSHA=head,runID='123',runAttempt=1,
                receiptRef=peak.receipt_ref('screen'),freezeSHA256=benchmark.digest(Path(tmp)/'freeze.json'),receiptObject='b'*40)
            with patch.dict(os.environ,dict(PAYLOAD_SHA=head,GITHUB_RUN_ID='123',GITHUB_RUN_ATTEMPT='1'),clear=True):
                peak.validate_receipt(receipt,'screen',head)
                with self.assertRaises(ValueError):peak.validate_receipt(receipt,'confirmation',head)
                for key,val in (('runID','124'),('payloadSHA','c'*40),('runAttempt',2),('freezeSHA256','d'*64)):
                    with self.assertRaises(ValueError):peak.validate_receipt(dict(receipt,**{key:val}),'screen',head)

    def test_host_binding_rejects_cross_boot_or_profile_before_stage(self):
        host=dict(stage='screen',bootID='one',workspace='/one',runID='123',runAttempt='1',profileSHA256='a'*64,
                  packageSHA256='b'*64,freezeSHA256='c'*64,toolingSHA256='d'*64)
        with tempfile.TemporaryDirectory() as tmp,patch.object(profile,'PROVENANCE',Path(tmp)):
            common.save(Path(tmp)/'screen-host.json',host)
            receipt=dict(hostIdentity=host,hostDocumentSHA256=benchmark.digest(Path(tmp)/'screen-host.json'))
            with patch.object(profile,'host_identity',return_value=host):self.assertEqual(host,profile.check_host('screen',receipt))
            for field in host:
                with patch.object(profile,'host_identity',return_value=dict(host,**{field:'different'})):
                    with self.assertRaises(ValueError):profile.check_host('screen',receipt)

    def test_control_bundle_uses_exact_original_v3_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=peak.control_bundle(Path(tmp))
            self.assertEqual(profile.CONTROL_JAR,benchmark.digest(folder/'Meridian.jar'))
            self.assertEqual('0.3.0',json.loads((folder/'bot/Meridian.json').read_text())['version'])

    def test_stage_driver_collects_all_valid_scores_but_stops_first_invalid(self):
        rows=[dict(id=f'fixture-{i}',stage='screen',arm='candidate') for i in range(5)]
        for invalid_at in (None,2):
            with self.subTest(invalid_at=invalid_at),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);reg=root/'reg';reg.mkdir();(reg/'freeze.json').write_text('{}')
                provenance=root/'provenance';provenance.mkdir()
                for name in ('screen-activation.json','package-contents.json','screen-host.json','prepared-environment.json'):(provenance/name).write_text('{}')
                calls=[]
                def play(game,*args):
                    calls.append(game['id'])
                    if invalid_at is not None and len(calls)==invalid_at:raise ValueError('synthetic invalid evidence')
                    row=scored_row(.01,0);del row['status'];return row
                with patch.object(peak,'OUT',root/'out'),patch.object(peak,'REG',reg),patch.object(peak,'PROVENANCE',provenance), \
                    patch.object(peak,'preflight',return_value={}),patch.object(peak,'validate_payload',return_value={}), \
                    patch.object(peak,'validate_receipt'),patch.object(peak,'check_host'),patch.object(peak,'control_bundle'), \
                    patch.object(peak,'verify_existing',return_value={}), \
                    patch.object(peak,'games',return_value=rows),patch.object(peak,'play',side_effect=play), \
                    patch.object(peak.subprocess,'check_output',return_value='a'*40+'\n'):
                    if invalid_at is None:peak.run('screen')
                    else:
                        with self.assertRaises(ValueError):peak.run('screen')
                self.assertEqual(5 if invalid_at is None else invalid_at,len(calls))

    def test_original_twelve_recordings_match_previous_official_aggregate_reducer(self):
        root=common.ROOT/'benchmarks/meridian-gun-plan'
        decision=json.loads((root/'decision.json').read_text())
        keys=('shots','hits','enemyShots','hitsTaken','damageDealt','damageTaken','shotEnergy','wallContacts','botContacts','skips','share','wins','turnsByRound','recordingSHA256')
        self.assertEqual(12,len(decision['games']))
        for original in decision['games']:
            game=dict(original,ownName='Meridian',ownVersion='0.3.0')
            result=common.metrics(root/'raw'/game['id'],game)
            for key in keys:self.assertEqual(original[key],result[key],(game['id'],key))


if __name__=='__main__':unittest.main()
