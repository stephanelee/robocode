#!/usr/bin/env python3
"""Render a recorded Tank Royale game to a silent H.264 MP4 (no battles)."""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, TURNS_PER_SECOND = 1280, 800, 20, 60
BG, PANEL, LINE, INK, MUTED = '#091522', '#10243a', '#2b425f', '#eef5ff', '#a3b6cf'
COLORS = ['#75bcff', '#f6c27b']
FONT_ROOT = Path('/System/Library/Fonts/Supplemental')


def font(size, bold=False):
    names = [FONT_ROOT / ('Arial Bold.ttf' if bold else 'Arial.ttf'),
             Path('/usr/share/fonts/truetype/dejavu') / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')]
    for name in names:
        if name.is_file():
            return ImageFont.truetype(str(name), size)
    return ImageFont.load_default(size=size)


def timestamp(seconds):
    millis = round(seconds * 1000)
    return f'{millis//3600000:02}:{millis//60000%60:02}:{millis//1000%60:02}.{millis%1000:03}'


def main(args):
    global TURNS_PER_SECOND
    TURNS_PER_SECOND = args.turns_per_second
    stage_label = args.stage_label or f'Swiss round {args.swiss_round}'
    if not 20 <= args.max_bitrate_kbps <= 130:
        raise ValueError('Video bitrate must be between 20 and 130 kbps')
    if args.output.exists():
        raise ValueError('Never overwrite a published video; choose a new output filename')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.recording, 'rt') as stream:
        first = json.loads(next(stream))
    if first.get('type') != 'GameStartedEventForObserver':
        raise ValueError('Expected the official observer recording format')
    setup = first['gameSetup']
    participants = sorted(first['participants'], key=lambda b: b['id'])
    if len(participants) != 2 or setup['numberOfRounds'] != 20:
        raise ValueError('This renderer expects the registered 20-round 1v1 game')
    ids = [b['id'] for b in participants]
    names = {b['id']: b['name'] for b in participants}
    colors = dict(zip(ids, COLORS))
    scores = {i: 0 for i in ids}
    places = {i: 0 for i in ids}
    through_round = 0
    ax, ay, aw, ah = 32, 128, 832, 624
    sx, sy = aw / setup['arenaWidth'], ah / setup['arenaHeight']
    fonts = {n: font(n) for n in [12, 14, 16, 18, 22, 28, 36]}
    bold = {n: font(n, True) for n in [18, 22, 28, 36, 46]}
    base = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(base)
    game_label = f'GAME {args.game} OF 2' if args.game < 3 else 'TIE-BREAK GAME 3'
    d.text((32, 22), f'ROBOCODE CUP  /  {stage_label.upper()}  /  {game_label}', font=fonts[14], fill='#75bcff')
    d.text((32, 49), names[ids[0]] + '  vs  ' + names[ids[1]], font=bold[28], fill=INK)
    d.text((32, 91), 'Recorded replay • all 20 battle rounds • accelerated playback', font=fonts[14], fill=MUTED)
    d.rounded_rectangle((ax-2, ay-2, ax+aw+2, ay+ah+2), radius=7, fill='#0c1c2c', outline=LINE, width=2)
    for x in range(50, setup['arenaWidth'], 50):
        px = ax + x*sx
        d.line((px, ay, px, ay+ah), fill='#142a3e')
    for y in range(50, setup['arenaHeight'], 50):
        py = ay + y*sy
        d.line((ax, py, ax+aw, py), fill='#142a3e')
    d.rounded_rectangle((896, 126, 1248, 752), radius=10, fill=PANEL, outline=LINE, width=1)
    d.text((32, 772), 'Accelerated replay • Energy updates throughout play • Official scores update after each round.', font=fonts[12], fill=MUTED)

    def world(x, y):
        return ax + x*sx, ay + (setup['arenaHeight']-y)*sy

    def polygon(cx, cy, angle, length, width, offset=0):
        dx, dy = math.cos(math.radians(angle)), -math.sin(math.radians(angle))
        nx, ny = -dy, dx
        cx += nx*offset*sx
        cy += ny*offset*sy
        return [(cx + dx*a*sx + nx*b*sx, cy + dy*a*sy + ny*b*sy)
                for a, b in [(-length/2,-width/2),(length/2,-width/2),(length/2,width/2),(-length/2,width/2)]]

    def frame(tick, message=None):
        image = base.copy()
        draw = ImageDraw.Draw(image)
        bots = {b['id']: b for b in tick.get('botStates', [])}
        for bullet in tick.get('bulletStates', []):
            x, y = world(bullet['x'], bullet['y'])
            color = colors.get(bullet['ownerId'], INK)
            a = math.radians(bullet['direction'])
            draw.line((x-10*math.cos(a), y+10*math.sin(a), x, y), fill=color, width=2)
            radius = 2 if bullet['power'] < 1 else 3
            draw.ellipse((x-radius,y-radius,x+radius,y+radius), fill=color)
        for identity, bot in bots.items():
            x, y = world(bot['x'], bot['y'])
            live = bot['energy'] > 0
            color = colors.get(identity, INK) if live else '#536275'
            heading = bot['direction']
            for offset in [-13, 13]:
                draw.polygon(polygon(x,y,heading,35,7,offset), fill='#040c14', outline=color, width=1)
            draw.polygon(polygon(x,y,heading,30,22), fill=color, outline='#d7e7f4', width=1)
            draw.ellipse((x-8,y-8,x+8,y+8), fill=PANEL, outline=color, width=2)
            gun = math.radians(bot['gunDirection'])
            draw.line((x,y,x+25*math.cos(gun),y-25*math.sin(gun)), fill='#040c14', width=6)
            draw.line((x,y,x+25*math.cos(gun),y-25*math.sin(gun)), fill=color, width=3)
            if not live:
                draw.line((x-10,y-10,x+10,y+10), fill=INK, width=2)
                draw.line((x-10,y+10,x+10,y-10), fill=INK, width=2)
        draw.text((920, 148), 'BATTLE ROUND', font=fonts[14], fill=MUTED)
        draw.text((920, 174), f"{tick.get('roundNumber',1):02d} / 20", font=bold[46], fill=INK)
        draw.text((920, 231), f"Turn {tick.get('turnNumber',0):,}  •  {TURNS_PER_SECOND} turns/sec", font=fonts[14], fill=MUTED)
        draw.text((920, 276), 'LIVE ENERGY', font=fonts[14], fill=MUTED)
        for number, identity in enumerate(ids):
            y = 307 + number*81
            energy = max(0, bots.get(identity, {}).get('energy', 100))
            draw.text((920,y), names[identity], font=bold[18], fill=colors[identity])
            draw.text((1224,y), f'{energy:.1f}', font=fonts[18], fill=INK, anchor='ra')
            draw.rounded_rectangle((920,y+30,1224,y+41), radius=4, fill='#071624')
            width = round(304*min(100,energy)/100)
            if width:
                draw.rounded_rectangle((920,y+30,920+width,y+41), radius=4, fill=colors[identity])
        draw.line((920,482,1224,482), fill=LINE)
        draw.text((920, 501), 'OFFICIAL SCORE', font=fonts[14], fill=MUTED)
        draw.text((920, 526), f'After completed round {through_round}', font=fonts[12], fill=MUTED)
        for number, identity in enumerate(ids):
            y = 558 + number*67
            draw.text((920,y), names[identity], font=fonts[16], fill=colors[identity])
            draw.text((1224,y-6), f'{scores[identity]:,}', font=bold[28], fill=INK, anchor='ra')
            draw.text((920,y+26), f'First places: {places[identity]}', font=fonts[12], fill=MUTED)
        draw.text((920, 713), 'Scores update at round end.', font=fonts[12], fill=MUTED)
        if message:
            draw.rounded_rectangle((150,376,746,493), radius=12, fill='#091522', outline='#75bcff', width=2)
            draw.text((448,405), message, font=bold[28], fill=INK, anchor='mm')
            draw.text((448,454), f"{names[ids[0]]} {scores[ids[0]]:,}  /  {names[ids[1]]} {scores[ids[1]]:,}", font=fonts[18], fill=MUTED, anchor='mm')
        return image

    temporary = args.output.with_name(args.output.stem+'.partial.mp4')
    command = ['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','25','-maxrate',f'{args.max_bitrate_kbps}k','-bufsize',f'{args.max_bitrate_kbps*4}k','-threads','2','-g',str(FPS*2),'-pix_fmt','yuv420p','-tag:v','avc1','-movflags','+faststart',str(temporary)]
    frames = 0
    chapters = []
    last = {'roundNumber':1,'turnNumber':0,'botStates':[],'bulletStates':[]}
    start_time = time.monotonic()
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(image, count=1):
        nonlocal frames
        raw = image.tobytes()
        for _ in range(count):
            process.stdin.write(raw)
            frames += 1

    try:
        step = TURNS_PER_SECOND // FPS
        seen_round = None
        with gzip.open(args.recording,'rt') as stream:
            for line in stream:
                row = json.loads(line)
                kind = row['type']
                if kind == 'TickEventForObserver':
                    last = row
                    if row['roundNumber'] != seen_round:
                        seen_round = row['roundNumber']
                        chapters.append(dict(round=seen_round,startSeconds=frames/FPS))
                        if seen_round == 1:
                            opening = frame(row, f'{stage_label} · Game {args.game}')
                            opening.save(args.output.with_suffix('.webp'), quality=90)
                            write(opening, FPS*2)
                        print(f'Rendering round {seen_round}/20', flush=True)
                    if row['turnNumber'] % step == 0 or row['turnNumber'] == 1:
                        write(frame(row))
                elif kind == 'RoundEndedEventForObserver':
                    # These arrays are already cumulative: replace, never sum them.
                    for bot in row['results']:
                        scores[bot['id']] = bot['totalScore']
                        places[bot['id']] = bot['firstPlaces']
                    through_round = row['roundNumber']
                    write(frame(last, f'Round {through_round} complete'), FPS//2)
                elif kind == 'GameEndedEventForObserver':
                    for bot in row['results']:
                        scores[bot['id']] = bot['totalScore']
                        places[bot['id']] = bot['firstPlaces']
        if seen_round != 20 or through_round != 20:
            raise ValueError('Replay did not cover all twenty rounds')
        if args.official:
            official = json.loads(args.official.read_text())
            assert scores == {b['id']: b['totalScore'] for b in official['results']}
        winner = max(ids, key=lambda identity:scores[identity])
        message = f'Game {args.game} ends in a draw' if len(set(scores.values())) == 1 else names[winner]+f' wins game {args.game}'
        write(frame(last, message), FPS*4)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('ffmpeg failed')
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    temporary.replace(args.output)
    duration = frames/FPS
    vtt = ['WEBVTT','']
    for number, chapter in enumerate(chapters):
        end = chapters[number+1]['startSeconds'] if number+1<len(chapters) else duration
        vtt.extend([f"{timestamp(chapter['startSeconds'])} --> {timestamp(end)}",f"Battle round {chapter['round']:02d}",''])
    args.output.with_suffix('.vtt').write_text('\n'.join(vtt))
    with args.recording.open('rb') as stream:
        source_sha = hashlib.file_digest(stream,'sha256').hexdigest()
    with args.output.open('rb') as stream:
        output_sha = hashlib.file_digest(stream,'sha256').hexdigest()
    report = dict(sourceRecording=args.recording.name,sourceSHA256=source_sha,videoSHA256=output_sha,videoSizeBytes=args.output.stat().st_size,width=W,height=H,fps=FPS,turnsPerSecond=TURNS_PER_SECOND,durationSeconds=duration,frames=frames,rounds=20,finalScores={names[i]:scores[i] for i in ids},chapters=chapters,elapsedRenderSeconds=round(time.monotonic()-start_time,2),note='Data-rendered replay, not a GUI screen recording. Official cumulative scores update at round end; no session IDs or private models are rendered.')
    report.update(stageLabel=stage_label, game=args.game, maxBitrateKbps=args.max_bitrate_kbps, rendererSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--official',type=Path)
    parser.add_argument('--turns-per-second',type=int,choices=[20,40,60,120,240],default=60)
    parser.add_argument('--swiss-round',type=int,default=4)
    parser.add_argument('--stage-label',help='Override the stage heading, for example Opening playoffs')
    parser.add_argument('--max-bitrate-kbps',type=int,default=130,help='20–130 kbps; the batch tool lowers this for longer movies')
    parser.add_argument('--game',type=int,choices=[1,2,3],default=1)
    main(parser.parse_args())
