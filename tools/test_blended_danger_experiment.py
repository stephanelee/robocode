import copy
import unittest
import blended_danger_experiment as e


def rows(stage):
    result=[]
    for game in e.layout(stage):
        previous={'diamond-port':.2,'adaptive-prime':.3,'Retcon':.7}.get(game['opponent'],.9)
        share=previous+(.1 if game['opponent'] in e.OPS[:2] else .05) if game['arm']=='candidate' else previous
        metrics=dict(turns=1000,shotsFired=20,bulletHits=3,damageDealt=20,damageTaken=30,wallCollisions=0,skippedTurns=0)
        result.append(dict(**game,battle=dict(rounds=game['rounds'],firstPlaces=game['rounds']//2+2,totalScore=1000*share,
            results=[dict(totalScore=1000*share),dict(totalScore=1000*(1-share))],metrics=metrics)))
    return result


class BlendedDangerGateTests(unittest.TestCase):
    def test_complete_stages_are60_then_conditional270_and_root_pins(self):
        self.assertEqual((60,240,30),tuple(sum(g['rounds'] for g in e.layout(s)) for s in ('screen','confirmation','samples')))
        self.assertEqual('advance',e.screen_decision(rows('screen'))['status'])
        self.assertEqual('retain',e.confirmation_decision(rows('confirmation')+rows('samples'))['status'])
        with self.assertRaises(ValueError):e.screen_decision(rows('screen')[:-1])

    def test_retcon_absolute_screen_share_and_wins_are_both_strict(self):
        data=rows('screen');ret=next(g for g in data if g['opponent']=='Retcon' and g['arm']=='candidate')
        ret['battle'].update(totalScore=500,results=[dict(totalScore=500),dict(totalScore=500)])
        self.assertFalse(e.screen_decision(data)['checks']['retconAbsolute'])
        data=rows('screen');next(g for g in data if g['opponent']=='Retcon' and g['arm']=='candidate')['battle']['firstPlaces']=5
        self.assertFalse(e.screen_decision(data)['checks']['retconAbsolute'])

    def test_confirm_each_advanced_block_and_retcon_majority_are_binding(self):
        data=rows('confirmation')+rows('samples')
        for g in data:
            if g['opponent']=='Retcon' and g['arm']=='candidate':g['battle']['firstPlaces']=10
        self.assertFalse(e.confirmation_decision(data)['checks']['retconAbsolute'])
        data=rows('confirmation')+rows('samples');bad=next(g for g in data if g['opponent']=='diamond-port' and g['arm']=='candidate')
        bad['battle'].update(totalScore=100,results=[dict(totalScore=100),dict(totalScore=900)])
        self.assertFalse(e.confirmation_decision(data)['checks']['advancedGains'])

    def test_relative_safety_and_candidate_only_sample_skips(self):
        data=rows('confirmation')+rows('samples')
        for g in data:
            if g['stage']=='samples' and g['arm']=='previous':g['battle']['metrics']['skippedTurns']=3
        self.assertTrue(e.confirmation_decision(data)['checks']['samples'])
        next(g for g in data if g['stage']=='samples' and g['arm']=='candidate')['battle']['metrics']['skippedTurns']=1
        self.assertFalse(e.confirmation_decision(data)['checks']['samples'])
        screen=rows('screen');next(g for g in screen if g['arm']=='candidate')['battle']['metrics']['wallCollisions']=11
        self.assertFalse(e.screen_decision(screen)['checks']['safety'])


if __name__=='__main__':unittest.main()
