"""Read-only endpoint/budget/timing audit. Imports no candidate or harness reducer."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import zipfile

ORIGINAL = Path(__file__).parents[1] / "ci-keelhelm-confirmation/run-41889609469/original.zip"
PREFIX = "robocode/robocode/build/keel-helm/confirmation/"
EXPECTED = "bcbd2e27ff64dc585f7341aa32ae506ea9819903c1893db8a723ba91c7036ad6"


class HashReader:
    def __init__(self, stream):
        self.stream, self.sha = stream, hashlib.sha256()

    def read(self, n=-1):
        data = self.stream.read(n)
        self.sha.update(data)
        return data


def main():
    with ORIGINAL.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == EXPECTED
    assert ORIGINAL.stat().st_size == 351880028
    rows = []
    with zipfile.ZipFile(ORIGINAL) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist())) == 2194
        decision = json.loads(archive.read(PREFIX + "decision.json"))
        population = json.loads(archive.read(PREFIX + "provenance/source/benchmarks/keel-helm-v1/games.json"))["confirmation"]
        assert len(population) == len(decision["games"]) == 18
        assert decision["unplayedIDs"] == []
        for game, recorded in zip(population, decision["games"]):
            assert all(recorded[k] == v for k, v in game.items()) and recorded["status"] == "valid"
            prefix = PREFIX + "games/" + game["id"] + "/"
            attempt = json.loads(archive.read(prefix + "attempt.json"))
            assert attempt["game"] == game and attempt["headSHA"] == "f2a25d3309062a362a13902329de405bf63094a7"
            assert attempt["freshProcess"] and attempt["freshWritableDirectory"] and not attempt["importedModels"]
            assert attempt["persistentKeelModelHashes"] is None
            recording, = [x for x in archive.infolist() if x.filename.startswith(prefix) and x.filename.endswith(".battle.gz")]
            raw = HashReader(archive.open(recording))
            ticks = collections.defaultdict(int)
            endpoints, start, end = [], None, None
            with gzip.GzipFile(fileobj=raw) as stream:
                for line in stream:
                    if b'"type":"TickEventForObserver"' in line[-50:]:
                        round_number = int(re.match(rb'\{"roundNumber":(\d+),', line).group(1))
                        position = line.rfind(b'"turnNumber":') + 13
                        turn = int(line[position:line.find(b",", position)])
                        assert turn == ticks[round_number] + 1
                        ticks[round_number] = turn
                    elif b"GameStartedEventForObserver" in line[-60:]:
                        start = json.loads(line)
                    elif b"RoundEndedEventForObserver" in line[-60:]:
                        endpoints.append(json.loads(line))
                    elif b"GameEndedEventForObserver" in line[-60:]:
                        end = json.loads(line)
            while raw.read(1 << 20):
                pass
            assert raw.sha.hexdigest() == recorded["recordingSHA256"]
            assert end["numberOfRounds"] == game["rounds"]
            setup = start["gameSetup"]
            assert (setup["gameType"], setup["arenaWidth"], setup["arenaHeight"], setup["turnTimeout"], setup["numberOfRounds"]) == ("1v1", 800, 600, 100000, game["rounds"])
            assert [x["roundNumber"] for x in endpoints] == list(range(1, game["rounds"] + 1))
            assert set(ticks) == set(range(1, game["rounds"] + 1))
            assert all(x["turnNumber"] == ticks[x["roundNumber"]] for x in endpoints)
            name = "Keel" if game["bot"] == "baseline" else "KeelHelm"
            opponent = "Diamond Port" if game["opponent"] == "diamond" else "Retcon"
            own, = [x for x in end["results"] if x["name"] == name]
            other, = [x for x in end["results"] if x["name"] == opponent]
            assert own["version"] == "0.1.0" and other["version"] == ("1.8.28" if game["opponent"] == "diamond" else "0.2.0")
            assert {x["name"]: x["totalScore"] for x in endpoints[-1]["results"]} == {x["name"]: x["totalScore"] for x in end["results"]}
            share = own["totalScore"] / (own["totalScore"] + other["totalScore"])
            assert share == recorded["share"] and own["totalScore"] == recorded["ownScore"] and own["firstPlaces"] == recorded["wins"]
            assert sum(ticks.values()) == recorded["turns"]
            markers = archive.read(prefix + "markers.txt").decode().splitlines()
            for tag in ("start", "end"):
                assert [int(x.split()[0]) for x in markers if x.endswith(" " + tag)] == list(range(1, game["rounds"] + 1))
            stats = [x for x in markers if " stats " in x]
            assert [int(x.split()[0]) for x in stats] == list(range(1, game["rounds"] + 1))
            maximum = max(int(re.search(r" maxMs=(\d+)", x).group(1)) for x in stats)
            assert maximum == recorded["maxMs"] < 75 and recorded["skips"] == sum(" skip " in x for x in markers) == 0
            rows.append(dict(game, share=share, firstPlaces=own["firstPlaces"], score=own["totalScore"], maxMs=maximum, skips=0, recordingSHA256=raw.sha.hexdigest()))
    group = lambda arm, opponent: [x for x in rows if x["bot"] == arm and x["opponent"] == opponent]
    means = {arm + "/" + opponent: sum(x["share"] for x in group(arm, opponent)) / len(group(arm, opponent)) for arm in ("baseline", "candidate") for opponent in ("diamond", "retcon")}
    candidate, baseline = means["candidate/diamond"], means["baseline/diamond"]
    retcon, retcon_baseline = means["candidate/retcon"], means["baseline/retcon"]
    checks = dict(allGamesValid=True, zeroBothArmSkips=True, bothArmTiming=True,
                  diamondAbove50=candidate > .5, diamondDeltaAtLeast2pp=candidate - baseline >= .02,
                  diamondWinsAbove60=sum(x["firstPlaces"] for x in group("candidate", "diamond")) > 60,
                  diamondGamesAbove50=sum(x["share"] > .5 for x in group("candidate", "diamond")) >= 4,
                  retconAtLeast90=retcon >= .9, retconDeltaAtLeastMinus2pp=retcon - retcon_baseline >= -.02,
                  retconWinsAtLeast24=sum(x["firstPlaces"] for x in group("candidate", "retcon")) >= 24)
    assert checks == decision["checks"] and decision["decision"] == "not confirmed"
    for actual, reported in ((candidate, decision["diamondMean"]), (candidate - baseline, decision["diamondDelta"]), (retcon, decision["retconMean"]), (retcon - retcon_baseline, decision["retconDelta"])):
        assert abs(actual - reported) < 1e-14  # arithmetic representation only; gate comparisons above are exact
    assert sum(x["rounds"] for x in rows) == decision["observedPhysicalRounds"] == decision["completedPhysicalRounds"] == 300
    result = dict(providerSHA256=EXPECTED, officialRounds=300, games=rows, means=means, checks=checks, decision="not confirmed")
    Path(__file__).with_name("original-endpoint-audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "games"}, indent=2))


if __name__ == "__main__":
    main()
