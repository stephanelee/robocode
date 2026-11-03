import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import coherent_duel_experiment as e


def battle(share, rounds, wins=2, walls=1):
    return dict(rounds=rounds, firstPlaces=wins, totalScore=1000*share,
        results=[dict(totalScore=1000*share),dict(totalScore=1000*(1-share))],
        metrics=dict(turns=1000,wallCollisions=walls,skippedTurns=0,shotsFired=10,bulletHits=3,damageDealt=10,damageTaken=10))


class CoherentExperimentTests(unittest.TestCase):
    def games(self):
        games=copy.deepcopy(e.layout())
        for game in games:
            game['battles']={a:battle(.6 if a=='candidate' else .4,game['rounds'],6 if game['kind']=='self' else 2) for a in game['arms']}
        return games

    def test_complete_registered_physical_and_perbot_budget(self):
        games=e.layout()
        self.assertEqual(len(games),15)
        self.assertEqual(sum(g['rounds'] for g in games),168)
        self.assertEqual(sum(g['rounds']*len(g['arms']) for g in games),198)
        self.assertEqual(sum(len(g['arms']) for g in games),18)
        self.assertEqual(e.decision(self.games())['status'],'retain')
        with self.assertRaises(ValueError): e.decision(self.games()[:-1])

    def test_all_gates_including_zero_wall_and_self_majority(self):
        games=self.games()
        for g in games:
            if g['kind']=='Retcon':
                g['battles'][g['arms'][0]]['metrics']['wallCollisions']=0
        self.assertTrue(e.decision(games)['checks']['wallRate'])
        games[1]['battles']['candidate']['metrics']['wallCollisions']=1
        self.assertFalse(e.decision(games)['checks']['wallRate'])
        for g in games:
            if g['kind']=='self':g['battles']['candidate']['firstPlaces']=5
        self.assertFalse(e.decision(games)['checks']['selfWins'])
        games[-1]['battles']['candidate']=battle(.34,3)
        self.assertFalse(e.decision(games)['checks']['samples'])

    def test_launchers_have_separate_model_metrics_seed_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); source=root/'source';source.mkdir()
            for name in e.ASSETS:(source/name).write_text('{}' if name.endswith('.json') else 'fixture')
            (source/'AISurvivor.json').write_text(json.dumps(dict(name='AISurvivor',version='1.2')))
            with patch.dict(e.JARS,dict(candidate=e.benchmark.digest(source/'AISurvivor.jar'),previous=e.benchmark.digest(source/'AISurvivor.jar'))):
                a=e.prepare_bot(source,root/'a','candidate',101,'/jdk/bin/java')
                b=e.prepare_bot(source,root/'b','previous',202,'/jdk/bin/java')
            self.assertNotEqual(a['name'],b['name'])
            self.assertNotEqual(a['model'],b['model']);self.assertNotEqual(a['metrics'],b['metrics'])
            self.assertIn('AISURVIVOR_SEED=101',(root/'a/AISurvivor.sh').read_text())
            self.assertIn('AISURVIVOR_SEED=202',(root/'b/AISurvivor.sh').read_text())
            self.assertIn('unset AISURVIVOR_DUEL_POLICY',(root/'a/AISurvivor.sh').read_text())
            self.assertEqual(e.clean_environment(dict(AISURVIVOR_MODEL_PATH='bad',AISURVIVOR_MODE='fixed',RETCON_STATS_DIR='bad',PATH='ok')),dict(PATH='ok'))

    @unittest.skipIf(os.name=='nt','Actual POSIX launchers execute on the Ubuntu match host')
    def test_actual_two_launchers_override_same_inherited_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();source=root/'source';source.mkdir()
            for name in e.ASSETS:(source/name).write_text('{}' if name.endswith('.json') else 'fixture')
            (source/'AISurvivor.json').write_text(json.dumps(dict(name='AISurvivor',version='1.2')))
            java=root/'java-probe'
            java.write_text('#!/bin/sh\nprintf "%s\\n" "$AISURVIVOR_MODE" "$AISURVIVOR_SEED" "$AISURVIVOR_MODEL_PATH" "$AISURVIVOR_METRICS_PATH" "${AISURVIVOR_DUEL_POLICY-unset}"\n')
            java.chmod(0o755)
            inherited=dict(os.environ,AISURVIVOR_MODE='eval',AISURVIVOR_SEED='0',AISURVIVOR_MODEL_PATH='shared',AISURVIVOR_METRICS_PATH='shared',AISURVIVOR_DUEL_POLICY='learned')
            with patch.dict(e.JARS,{a:e.benchmark.digest(source/'AISurvivor.jar') for a in ('candidate','previous')}):
                for arm,seed in (('candidate',101),('previous',202)):
                    bot=e.prepare_bot(source,root/arm,arm,seed,str(java))
                    actual=subprocess.check_output(['/bin/sh',str(root/arm/'AISurvivor.sh')],env=inherited,text=True).splitlines()
                    self.assertEqual(actual,['train',str(seed),bot['model'],bot['metrics'],'unset'])


if __name__=='__main__':unittest.main()
