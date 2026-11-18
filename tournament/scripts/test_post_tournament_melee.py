import copy
import json
from pathlib import Path
import unittest
import post_tournament_melee as melee

class MeleeTests(unittest.TestCase):
    def fixture(self):
        names=["KeelHelm","Diamond Port","Keel","Prism"]
        data=dict(numberOfRounds=20,results=[dict(id=i+1,name=n,version="test",rank=i+1,totalScore=100-i*10,firstPlaces=5,bulletDamage=20,survival=10) for i,n in enumerate(names)])
        return names,data,[(n,"test") for n in names]
    def test_finish_order_comes_from_bracket_not_swiss_seeding(self):
        base=Path(__file__).resolve().parents[2]/"benchmarks/tournament-2026-11-04"
        before={p.name:p.read_bytes() for p in base.glob("playoff-*.json")}
        self.assertEqual(["KeelHelm","Diamond Port","Keel","Prism"],melee.top_four(base))
        self.assertEqual(before,{p.name:p.read_bytes() for p in base.glob("playoff-*.json")})
    def test_four_complete_results_are_required(self):
        names,data,ids=self.fixture();melee.validate_result(data,ids)
        for mutation in ("missing","rounds","identity","firstPlaces"):
            bad=copy.deepcopy(data)
            if mutation=="missing":bad["results"].pop()
            if mutation=="rounds":bad["numberOfRounds"]=19
            if mutation=="identity":bad["results"][0]["version"]="other"
            if mutation=="firstPlaces":bad["results"][0]["firstPlaces"]=21
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                melee.validate_result(bad,ids)
    def test_duplicate_ids_and_negative_scores_rejected(self):
        names,data,ids=self.fixture()
        for key,value in (("id",2),("totalScore",-1),("rank",1.5)):
            bad=copy.deepcopy(data);bad["results"][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):melee.validate_result(bad,ids)
    def test_combined_scores_use_names_not_launch_order(self):
        names,data,ids=self.fixture();other=copy.deepcopy(data);other["results"].reverse()
        report=dict(entrants=names,games=[dict(status="complete",official=g) for g in (data,other)])
        rows=melee.totals(report)
        self.assertEqual([200,180,160,140],[r["totalScore"] for r in rows])
        self.assertEqual([1,2,3,4],[r["rank"] for r in rows])
    def test_exact_exhibition_ties_share_rank_without_extra_games(self):
        names,data,ids=self.fixture()
        for row in data["results"]:row.update(totalScore=100,firstPlaces=5)
        report=dict(entrants=names,games=[dict(status="complete",official=data)])
        self.assertEqual([1,1,1,1],[r["rank"] for r in melee.totals(report)])
if __name__=="__main__":unittest.main()
