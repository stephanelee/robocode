"""Verify all original payloads and reproduce the frozen V2 rejection offline; zero games."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source/tools'))
import ais_breakwater_selective_transfer as frozen


def main(extract=None):
    mapping = json.loads((ROOT / 'original-files.json').read_bytes())
    provenance = json.loads((ROOT / 'provenance.json').read_bytes())
    with zipfile.ZipFile(ROOT / 'original-payloads.zip') as archive:
        assert set(archive.namelist()) == {v['payload'] for v in mapping.values()}
        payloads = {name: archive.read(name) for name in archive.namelist()}
    for name, entry in mapping.items():
        path = PurePosixPath(name)
        assert not path.is_absolute() and '..' not in path.parts and '\\' not in name
        data = payloads[entry['payload']]
        assert hashlib.sha256(data).hexdigest() == entry['sha256'] and len(data) == entry['bytes'], name
        if extract is not None:
            output = extract / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
    def raw(name): return payloads[mapping[name]['payload']]
    report = json.loads(raw('experiment.json'))
    assert (ROOT / 'experiment.json').read_bytes() == raw('experiment.json')
    assert report['status'] == 'complete' and len(report['games']) == 9
    assert len(mapping) == provenance['originalFileCount'] == 956
    assert sum(g['rounds'] for g in report['games']) == 90
    assert report['runtime']['hashes'] == json.loads(raw('registered-protocol.json'))['runtime']['hashes']
    assert report['registration'] == frozen.REGISTRATION and report['sources'] == dict(previous=frozen.BASE,candidate=frozen.SOURCE)
    assert report['jarSHA256'] == frozen.JARS
    assert hashlib.sha256(raw('registered-plan.md')).hexdigest() == report['planSHA256']
    assert hashlib.sha256(raw('registered-protocol.json')).hexdigest() == report['protocolSHA256']
    assert frozen.decision(report['games']) == report['screenDecision']
    assert report['decision'] == dict(status='reject-at-screen', confirmationRounds=0)
    models, rows_count = set(), 0
    for game in report['games']:
        prefix = game['gameId']
        official = json.loads(raw(prefix + '/battle.json'))
        assert official['numberOfRounds'] == game['rounds']
        for name,sha in game['pythonSourceHashes'].items():
            assert hashlib.sha256(raw(prefix+'/'+name)).hexdigest() == sha
        if game['retconStats']:
            assert game['retconStats']['data']['roundsRecorded'] == game['rounds']
            for item in game['retconStats']['rawArtifacts']:
                assert hashlib.sha256(raw(prefix+'/retcon-stats/'+item['path'])).hexdigest() == item['sha256']
        for arm, battle in game['battles'].items():
            base = prefix + '/' + arm + '/AISurvivor/'
            rows = [json.loads(line) for line in raw(base + 'metrics.jsonl').splitlines()]
            frozen.benchmark.validate_metrics(rows, game['rounds'], 'train', game['seeds'][arm], game['rounds'])
            assert frozen.benchmark.summarize_metrics(rows) == battle['metrics']
            assert frozen.policy.mechanism(rows) == battle['mechanism']
            assert frozen.movement_activation(rows, arm) == battle['breakwaterMovementTurns']
            assert frozen.selectivity_activation(rows, arm) == battle['selectivity']
            assert hashlib.sha256(raw(base + 'AISurvivor.jar')).hexdigest() == frozen.JARS[arm]
            for name,sha in battle['bot']['launchedHashes'].items():
                assert hashlib.sha256(raw(base+name)).hexdigest() == sha
            model = raw(base + 'model.dat')
            assert hashlib.sha256(model).hexdigest() == battle['endpoint']['sha256']
            header = struct.unpack('>7i', model[:28])
            assert header[:5] == (0x41495356, 1, 14, 32, 6) and header[5] == game['rounds'] and header[6] >= 0
            assert all(math.isfinite(v) for v in struct.unpack('>' + str((len(model) - 28) // 8) + 'd', model[28:]))
            models.add(base + 'model.dat')
            rows_count += len(rows)
            identity = next(r for r in official['results'] if (r['name'], r['version']) == (battle['name'], battle['version']))
            assert all(identity[k] == battle[k] for k in identity)
        if game['kind'] == 'Breakwater':
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                for name in ('breakwater-model.json', 'breakwater.jsonl'):
                    (path / name).write_bytes(raw(prefix + '/' + name))
                opponent = next(r for r in official['results'] if r['name'] == 'Breakwater')
                assert frozen.check_breakwater(path, game['rounds'], game['opponentSeed'], opponent) == game['breakwater']
    assert len(models) == 10 and rows_count == 100
    print('Verified956 original files/550 unique payloads,90 physical rounds/100 AIS rows/10 cold models; frozen V2 rejection and selective-fire counters reproduced. Zero games.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract', type=Path, help='Restore all original paths into a NEW directory')
    args = parser.parse_args()
    if args.extract is not None:
        args.extract = args.extract.resolve()
        args.extract.mkdir(parents=True, exist_ok=False)
    main(args.extract)
