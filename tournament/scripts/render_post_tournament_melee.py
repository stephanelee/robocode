"""Render verified exhibition recordings and merge separate replay metadata; no games."""
import argparse,concurrent.futures,gzip,hashlib,json,shutil,subprocess,sys
from pathlib import Path
import tournament

def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def main(args):
    report=json.loads((args.output/"report.json").read_text())
    if report["status"]!="complete" or report["completedGames"]!=2:
        raise ValueError("Require both validated exhibition games")
    args.media_dir.mkdir(parents=True,exist_ok=True)
    def render(game):
        recording=args.output/game["recording"]["path"];official=args.output/game["directory"]/"battle.json"
        if sha(recording)!=game["recording"]["sha256"] or json.loads(official.read_text())!=game["official"]:
            raise ValueError("Original exhibition evidence changed")
        movie=args.media_dir/f"post-melee-g{game['game']}-slow-v1.mp4";meta=movie.with_suffix(".json")
        if not meta.exists():
            if movie.exists():raise ValueError("Unverified existing movie")
            with gzip.open(recording,"rb") as stream:ticks=sum(b'"TickEventForObserver"' in line for line in stream)
            bitrate=min(130,max(20,int(21_000_000*8/(ticks/60+30)/1000)))
            with movie.with_suffix(".log").open("x") as log:
                subprocess.run([sys.executable,str(Path(__file__).with_name("render_tournament_replay.py")),"--recording",str(recording),"--official",str(official),"--stage-label","Post-tournament melee","--game",str(game["game"]),"--max-bitrate-kbps",str(bitrate),"--output",str(movie)],stdout=log,stderr=subprocess.STDOUT,check=True)
        proof=json.loads(meta.read_text())
        if proof["sourceSHA256"]!=game["recording"]["sha256"] or proof["videoSHA256"]!=sha(movie) or proof["finalScores"]!={r["name"]:r["totalScore"] for r in game["official"]["results"]} or [c["round"] for c in proof["chapters"]]!=list(range(1,21)):
            raise ValueError("Exhibition video provenance differs")
        if movie.stat().st_size>=24*1024**2:raise ValueError("Movie too large")
        probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration:stream=codec_name,pix_fmt,width,height,nb_frames","-of","json",str(movie)]))
        stream=probe["streams"][0]
        if (stream["codec_name"],stream["pix_fmt"],stream["width"],stream["height"],int(stream["nb_frames"]))!=("h264","yuv420p",1280,800,proof["frames"]) or abs(float(probe["format"]["duration"])-proof["durationSeconds"])>.1:raise ValueError("Video encoding differs")
        return game,movie,proof
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rendered=list(pool.map(render,report["games"]))
    public_size=sum(p.stat().st_size for p in args.site.rglob("*") if p.is_file())
    if public_size+sum(p["videoSizeBytes"] for g,m,p in rendered)+200_000>256*1024**2:
        subprocess.run([sys.executable,str(Path(__file__).with_name("render_tournament_replays.py")),"--output",str(args.tournament),"--media-dir",str(args.media_dir),"--site",str(args.site),"--omit-swiss4","--archive-lower-before-wave","6"],check=True)
    videos=[];target=args.site/"videos";target.mkdir(exist_ok=True)
    for game,movie,proof in rendered:
        pieces=[]
        if movie.stat().st_size>8_000_000:
            with movie.open("rb") as stream:
                n=0
                while b:=stream.read(4_000_000):
                    n+=1;name=movie.stem+f".part{n:02d}.bin";p=target/name;digest=hashlib.sha256(b).hexdigest()
                    if p.exists():
                        if sha(p)!=digest:raise ValueError("Existing piece differs")
                    else:p.write_bytes(b)
                    pieces.append(dict(url="videos/"+name,sizeBytes=len(b),sha256=digest))
        for suffix in ([".webp",".vtt"] if pieces else [".mp4",".webp",".vtt"]):
            src=movie.with_suffix(suffix);dst=target/src.name
            if dst.exists():
                if sha(dst)!=sha(src):raise ValueError("Existing public asset differs")
            else:shutil.copy2(src,dst)
        seconds=round(proof["durationSeconds"])
        item=dict(stage="exhibition",game=game["game"],title=f"Top-four melee · Game {game['game']}",label=f"Post-tournament melee · 20 rounds · {seconds//60}:{seconds%60:02d} at 1×",fileName=movie.name,mp4=None if pieces else "videos/"+movie.name,poster="videos/"+movie.with_suffix(".webp").name,chaptersFile="videos/"+movie.with_suffix(".vtt").name,chapters=proof["chapters"],sha256=proof["videoSHA256"],sizeBytes=proof["videoSizeBytes"],turnsPerSecond=60,participants=game["participants"])
        if pieces:item["parts"]=pieces
        videos.append(item)
    data=json.loads((args.site/"data.json").read_text());data["exhibition"]["videos"]=videos;data["updated"]=tournament.stamp();tournament.save(args.site/"data.json",data)
    tournament.save(args.media_dir/"post-melee-video-verification.json",dict(games=2,rounds=40,newBattles=0,videos=[p for g,m,p in rendered],publicVideos=videos))
    print("Both four-bot exhibition replays verified and exported",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("output","tournament","media-dir","site"):p.add_argument("--"+name,type=Path,required=True)
    main(p.parse_args())
