"""One exclusive persisted balanced draw for issue138; no engine or game capability."""
from pathlib import Path
import hashlib,json,math,secrets,sys,os
from datetime import datetime,timezone
sys.path.insert(0,'/private/tmp/robocode-crossfire-d36d83a/tournament/tools')
from keelhelm_randomization import unrank
p=Path('/Users/robert/.codex/artifacts/robocode-keelhelm-crossfire/2026-11-18/assignment.json')
# Exclusive creation comes BEFORE randomness: a pre-existing assignment can never consume a new draw.
with p.open('x',encoding='utf-8')as stream:
 rank=secrets.randbelow(math.comb(32,16))
 value=dict(study='keelhelm-crossfire-138',mechanism='one secrets.randbelow(C(32,16)); lexicographic unranking',rank=rank,space=math.comb(32,16),candidateFirst=unrank(32,16,rank),createdUTC=datetime.now(timezone.utc).isoformat(),sourceSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps(value))
