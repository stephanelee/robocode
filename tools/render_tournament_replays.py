"""Render all completed recorded games from Swiss 4 onward; publish local media/catalog.

No battles or network access. Run in the separate video Python venv, then publish
the resulting Site source with the Sites workflow. Existing movies are verified
and reused without overwriting them. Raw recordings never enter the public site.
"""
import argparse
import concurrent.futures
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def stage_label(stage):
    if stage == 'swiss-4':
        return 'Swiss round 4'
    number = int(stage.split('-')[1])
    return {1:'Opening playoffs', 2:'Playoffs wave 2', 3:'Playoffs wave 3',
            4:'Playoffs wave 4', 5:'Lower final', 6:'Grand final', 7:'Grand-final reset'}[number]


def jobs(output):
    reports = [output/'swiss-4.json'] + sorted(p for p in output.glob('playoff-*.json') if re.fullmatch(r'playoff-[1-7]\.json', p.name))
    result = []
    for path in reports:
        report = json.loads(path.read_text())
        for pair_number, pair in enumerate(report['draw'], 1):
            for game in pair['games']:
                if game['status'] != 'complete':
                    continue
                if not game.get('recording'):
                    raise ValueError('Completed game lacks a recording: '+game['directory'])
                result.append((report['stage'], pair_number, pair, game))
    return result


def main(args):
    args.output = args.output.resolve()
    args.media_dir = args.media_dir.resolve()
    args.media_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.site/'data.json'
    data = json.loads(data_path.read_text())
    original_updated = data['updated']
    existing = data.get('videos', [])
    renderer = Path(__file__).with_name('render_tournament_replay.py')

    def split_for_transfer(movie, expected_sha):
        if movie.stat().st_size <= args.chunk_bytes*2:
            return []
        result=[];whole=hashlib.sha256()
        with movie.open('rb') as stream:
            number=0
            while content:=stream.read(args.chunk_bytes):
                whole.update(content);number+=1
                name=movie.stem+f'.part{number:02d}.bin'
                target=args.site/'videos'/name
                expected=hashlib.sha256(content).hexdigest()
                if target.exists():
                    if sha(target)!=expected:raise ValueError('Existing video part differs: '+name)
                else:
                    with target.open('xb') as output:output.write(content)
                result.append(dict(url='videos/'+name,sizeBytes=len(content),sha256=expected))
        if whole.hexdigest()!=expected_sha:raise ValueError('Transfer pieces differ from original video')
        return result

    def render(item):
        stage, pair_number, pair, game = item
        title = f'{pair["a"]} vs {pair["b"]} · Game {game["game"]}'
        prior = next((v for v in existing if (v.get('stage') == stage and v.get('matchA') == pair['a'] and v.get('matchB') == pair['b'] and v.get('game') == game['game']) or (stage == 'swiss-4' and v['title'] == title)), None)
        name = Path(prior.get('fileName') or prior['mp4']).stem if prior else f'{stage}-p{pair_number:02d}-g{game["game"]}-slow-{args.edition}'
        if not re.fullmatch(r'[a-zA-Z0-9-]+', name):
            raise ValueError('Unsafe media basename')
        movie = args.media_dir/(name+'.mp4')
        meta = movie.with_suffix('.json')
        recording = args.output/game['recording']['path']
        official = args.output/game['directory']/'battle.json'
        if sha(recording) != game['recording']['sha256']:
            raise ValueError('Source recording hash differs: '+game['directory'])
        if json.loads(official.read_text()) != game['official']:
            raise ValueError('Official result differs: '+game['directory'])
        if not meta.exists():
            if movie.exists():
                raise ValueError('Unverified movie exists; retain it and choose a new --edition')
            with gzip.open(recording, 'rb') as stream:
                ticks = sum(b'"TickEventForObserver"' in line for line in stream)
            bitrate = min(130, max(20, int(21_000_000*8/(ticks/60+30)/1000)))
            command = [sys.executable, str(renderer), '--recording', str(recording),
                       '--official', str(official), '--stage-label', stage_label(stage),
                       '--game', str(game['game']), '--turns-per-second', '60',
                       '--max-bitrate-kbps', str(bitrate), '--output', str(movie)]
            with movie.with_suffix('.log').open('x') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads(meta.read_text())
        if report['sourceSHA256'] != game['recording']['sha256'] or report['videoSHA256'] != sha(movie):
            raise ValueError('Video/source provenance differs: '+name)
        expected = {r['name']:r['totalScore'] for r in game['official']['results']}
        if report['finalScores'] != expected or [c['round'] for c in report['chapters']] != list(range(1,21)):
            raise ValueError('Video score or chapter coverage differs: '+name)
        if movie.stat().st_size != report['videoSizeBytes'] or movie.stat().st_size >= 24*1024*1024:
            raise ValueError('Video exceeds the per-asset budget or size differs: '+name)
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
                           'format=duration:stream=codec_name,pix_fmt,width,height,nb_frames','-of','json',str(movie)]))
        stream = probe['streams'][0]
        if (stream['codec_name'],stream['pix_fmt'],stream['width'],stream['height']) != ('h264','yuv420p',1280,800):
            raise ValueError('Unexpected video encoding: '+name)
        if int(stream['nb_frames']) != report['frames'] or abs(float(probe['format']['duration'])-report['durationSeconds']) > .1:
            raise ValueError('Video frame/duration metadata differs: '+name)
        (args.site/'videos').mkdir(parents=True,exist_ok=True)
        pieces=split_for_transfer(movie, report['videoSHA256'])
        for extension in (['.webp','.vtt'] if pieces else ['.mp4','.webp','.vtt']):
            local = movie.with_suffix(extension)
            target = args.site/'videos'/local.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if sha(target) != sha(local):
                    raise ValueError('Refusing to replace an existing public asset: '+str(target))
            else:
                shutil.copy2(local, target)
        seconds = round(report['durationSeconds'])
        entry = dict(stage=stage, matchA=pair['a'], matchB=pair['b'], game=game['game'],
                     title=title, label=f'{stage_label(stage)} · 20 rounds · {seconds//60}:{seconds%60:02d} at 1×',
                     mp4='videos/'+movie.name, poster='videos/'+movie.with_suffix('.webp').name,
                     chaptersFile='videos/'+movie.with_suffix('.vtt').name, chapters=report['chapters'],
                     sizeBytes=report['videoSizeBytes'], turnsPerSecond=report['turnsPerSecond'])
        entry['fileName']=movie.name
        if pieces:
            entry['parts']=pieces;entry['mp4']=None
            duplicate=args.site/'videos'/movie.name
            if duplicate.exists():
                if duplicate.resolve()==movie.resolve() or sha(duplicate)!=report['videoSHA256']:raise ValueError('Refusing to remove an unverified duplicate')
                duplicate.unlink()
        entry['sha256']=report['videoSHA256']
        proof = dict(stage=stage, game=game['directory'], sourceSHA256=report['sourceSHA256'],
                     videoSHA256=report['videoSHA256'], sizeBytes=report['videoSizeBytes'],
                     durationSeconds=report['durationSeconds'], chaptersVerified=20,
                     codecVerified=True, finalScores=report['finalScores'])
        if pieces:proof['parts']=pieces
        print('Verified '+game['directory'], flush=True)
        return entry, proof

    selected_jobs = jobs(args.output)
    older_lower = {(stage, pair['a'], pair['b'], game['game'])
                   for stage, _, pair, game in selected_jobs
                   if stage.startswith('playoff-')
                   and int(stage.split('-')[1]) < args.archive_lower_before_wave
                   and pair.get('matchId', '').startswith('L')}
    selected_jobs = [item for item in selected_jobs
                     if (item[0], item[2]['a'], item[2]['b'], item[3]['game']) not in older_lower]
    if args.omit_swiss4 or older_lower:
        selected_jobs = [item for item in selected_jobs if item[0] != 'swiss-4']
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        rendered = list(pool.map(render, selected_jobs))
    if json.loads(data_path.read_text())['updated'] != original_updated:
        raise ValueError('Site data changed during rendering; rerun to merge into the newer snapshot')
    entries, verification = zip(*rendered)
    data['videos'] = list(entries)
    data['video'] = entries[0]
    data['replayPolicy'] = dict(fromStage='swiss-4', gamesAvailable=len(entries),
                                battleRoundsAvailable=20*len(entries), defaultPlaybackRate=.25, maxTransferPartBytes=args.chunk_bytes)
    if args.omit_swiss4:
        # Remove only verified, reproducible public copies. Private movies and raw
        # recordings remain intact; an unrelated file or symlink blocks cleanup.
        archived = [v for v in existing if (args.omit_swiss4 and v.get('stage') == 'swiss-4')
                    or (v.get('stage'), v.get('matchA'), v.get('matchB'), v.get('game')) in older_lower]
        removals = []
        for entry in archived:
            for url in [entry.get('mp4'), entry['poster'], entry['chaptersFile']]:
                if url:
                    target = args.site/url
                    original = args.media_dir/Path(url).name
                    if target.exists():
                        if target.is_symlink() or target.resolve() == original.resolve() or sha(target) != sha(original):
                            raise ValueError('Refusing to remove an unverified public copy: '+url)
                        removals.append(target)
            for part in entry.get('parts', []):
                target = args.site/part['url']
                if target.exists():
                    if target.is_symlink() or sha(target) != part['sha256']:
                        raise ValueError('Refusing to remove an unverified public part')
                    removals.append(target)
        archive = args.media_dir/(f'lower-before-wave{args.archive_lower_before_wave}-public-catalog-archived.json'
                                  if older_lower else 'swiss-4-public-catalog-archived.json')
        if archived and not archive.exists():
            save(archive, archived)
        for target in removals:
            target.unlink()
        data['replayPolicy']['archivedStages'] = ['swiss-4'] if args.omit_swiss4 else []
        data['replayPolicy']['archivedGames'] = [dict(stage=s, matchA=a, matchB=b, game=g)
                                                for s,a,b,g in sorted(older_lower)]
    data['updated'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(data_path, data)
    proof = dict(gamesVerified=len(entries), battleRoundsVerified=20*len(entries), newBattles=0,
                 totalBytes=sum(p['sizeBytes'] for p in verification), games=list(verification))
    save(args.media_dir/'replay-catalog-verification.json', proof)
    print(json.dumps({k:v for k,v in proof.items() if k!='games'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Canonical tournament evidence directory')
    parser.add_argument('--media-dir', type=Path, required=True, help='Private video output/provenance directory')
    parser.add_argument('--site', type=Path, required=True, help='Initialized public static directory')
    parser.add_argument('--workers', type=int, choices=[1,2], default=2)
    parser.add_argument('--edition', default='v1', help='Immutable filename edition for new renders')
    parser.add_argument('--omit-swiss4', action='store_true', help='Archive Round 4 public videos to make room; keep private movies and recordings')
    parser.add_argument('--archive-lower-before-wave', type=int, choices=range(0,9), default=0,
                        help='Archive only lower-bracket public videos from earlier waves; preserve all private evidence')
    parser.add_argument('--chunk-bytes',type=int,choices=[4_000_000],default=4_000_000,help='Lossless transport piece size for movies larger than8MB')
    main(parser.parse_args())
