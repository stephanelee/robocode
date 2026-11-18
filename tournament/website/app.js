'use strict';
let D,kind='All',query='',profileId=null,activeStage=new URLSearchParams(location.search).get('stage');
let selectedReplay=null,playAllReplays=false,replayLoadId=0,replayAbort=null,lastRenderedStage=null;
const page=document.body.dataset.page,$=s=>document.querySelector(s),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sameName=(a,b)=>String(a).replace(/\s/g,'')===String(b).replace(/\s/g,'');
const waveLabel=n=>({1:'Opening playoffs',2:'Upper semifinals & Lower round 1',3:'Lower round 2',4:'Upper final & Lower round 3',5:'Lower final',6:'Grand final',7:'Grand-final reset'}[n]||'Playoffs');
const stageLabel=s=>s==='exhibition'?'Post-tournament melee':s==='practice'?'Practice':s.startsWith('swiss-')?'Swiss '+s.split('-')[1]:waveLabel(Number(s.split('-')[1]));
const replayItems=()=>[...(D.videos?.length?D.videos:D.video?[D.video]:[]),...(D.exhibition?.videos||[])];
function replayScope(){return replayItems().flatMap((v,i)=>(v.stage||(v.label?.startsWith('Swiss round 4')?'swiss-4':null))===activeStage?[i]:[])}
function stages(){return [{stage:'practice',draw:D.practice||[],status:'complete',plannedGames:14,completedGames:D.completedGames||0},...(D.rounds||[]),...(D.playoffs||[])]}
function latestStage(){return stages().at(-1)}
function resultFor(game,name){return game.official?.results.find(r=>sameName(r.name,name))}
function videoIndex(stage,pair,game){return replayItems().findIndex(v=>v.stage===stage&&v.matchA===pair.a&&v.matchB===pair.b&&v.game===game||stage==='swiss-4'&&v.title===`${pair.a} vs ${pair.b} · Game ${game}`)}
function roster(){const list=D.profiles.filter(b=>(kind==='All'||(kind==='Home'?['Home','Experimental'].includes(b.kind):b.kind===kind))&&(b.name+' '+b.description).toLowerCase().includes(query));$('#roster').innerHTML=list.map(b=>`<button class="card" data-kind="${esc(b.kind)}" data-id="${esc(b.slug)}" aria-label="Read ${esc(b.name)} profile"><div class="card-top"><span class="avatar">${esc(b.name.replace(/[^A-Z]/g,'').slice(0,2)||b.name.slice(0,2).toUpperCase())}</span><span class="pill">${esc(b.kind)}</span></div><div><h3>${botName(b.name,false)}</h3><p>${esc(b.tagline)}</p></div><p>${esc(b.description)}</p><div class="card-foot"><span>v${esc(b.version)}</span><span>View profile ↗</span></div></button>`).join('');$('#noresults').hidden=!!list.length;document.querySelectorAll('.card').forEach(c=>c.onclick=()=>profile(c.dataset.id))}function botName(name,badge=true){const guest=D.profiles.some(b=>b.name===name&&b.kind==='Guest');return guest?`<span class="guest-name">${esc(name)}</span>${badge?'<small class="guest-tag">GUEST</small>':''}`:esc(name)}function profileResults(name){const rows=[];const add=(pair,label,ranked)=>{if(!pair)return;const isA=pair.a===name,opponent=isA?pair.b:pair.a,done=pair.status==='complete';const mine=isA?pair.scoreA:pair.scoreB,other=isA?pair.scoreB:pair.scoreA;const outcome=done?(pair.winner?(pair.winner===name?'W':'L'):(mine>other?'W':mine<other?'L':'D')):'…';const score=done?Number(mine).toLocaleString('en-US')+'–'+Number(other).toLocaleString('en-US'):'—';const points=done&&ranked?(isA?pair.pointsA:pair.pointsB):'—';rows.push(`<tr class="profile-result-${outcome==='W'?'win':outcome==='L'?'loss':'pending'}"><td><span class="profile-round">${esc(label)}</span><small class="profile-outcome" title="${outcome==='W'?'Win':outcome==='L'?'Loss':outcome==='D'?'Draw':'Pending'}">${outcome}</small></td><td>${botName(opponent,false)}</td><td class="profile-score">${score}</td><td class="profile-points">${points}</td></tr>`)};add(D.practice.find(p=>p.a===name||p.b===name),'Practice',false);for(const round of D.rounds||[])add(round.draw.find(p=>p.a===name||p.b===name),'R'+round.stage.split('-')[1],true);for(const wave of D.playoffs||[]){const pair=wave.draw.find(p=>p.a===name||p.b===name);add(pair,pair?.matchId||'Playoff',false)}const standing=(D.standings||[]).find(r=>r.name===name);return `<section class="profile-results" aria-label="Tournament results"><div class="profile-results-title"><h3>Tournament results</h3><span>${standing?standing.points:0} Swiss pts</span></div><table><colgroup><col style="width:20%"><col style="width:34%"><col style="width:34%"><col style="width:12%"></colgroup><thead><tr><th>Round</th><th>Opponent</th><th class="profile-score">Score</th><th class="profile-points">Pts</th></tr></thead><tbody>${rows.join('')}</tbody></table><p class="profile-results-note">Scores combine the pairing’s games. Practice is unranked; playoffs award advancement rather than Swiss points.</p></section>`}function profile(id){profileId=id;const b=D.profiles.find(x=>x.slug===id);$('#profilebody').innerHTML=`<span class="pill green">${esc(b.kind)}</span><h2 id="profile-title">${botName(b.name,false)}</h2><div class="eyebrow">${esc(b.tagline)}</div><p>${esc(b.description)}</p><dl><dt>Version</dt><dd>${esc(b.version)}</dd><dt>Targeting</dt><dd>${esc(b.gun)}</dd><dt>Movement</dt><dd>${esc(b.movement)}</dd><dt>Runtime</dt><dd>${esc(b.runtime)}</dd></dl>${profileResults(b.name)}<a class="primary" style="display:inline-block;margin-top:10px" href="${esc(b.source)}" rel="nofollow noopener" target="_blank">View pinned source ↗</a>`;if(!$('#profile').open)$('#profile').showModal()}
function sumScores(pair){if(pair.scoreA!=null)return [pair.scoreA,pair.scoreB];return [pair.a,pair.b].map(name=>(pair.games||[]).filter(g=>g.status==='complete').reduce((sum,g)=>sum+(resultFor(g,name)?.totalScore||0),0))}
function matchCard(pair,stage){
 const complete=pair.status==='complete',scores=sumScores(pair),games=pair.games||[],done=games.filter(g=>g.status==='complete').length;
 const winner=complete?(pair.winner||(scores[0]>scores[1]?pair.a:scores[1]>scores[0]?pair.b:null)):null;
 const entrant=(name,i)=>`<div class="${i?'pair-b':''}"><strong class="${winner===name?'pair-winner':''}">${botName(name,false)}</strong><span class="pair-total">${done?scores[i].toLocaleString('en-US'):'—'}</span>${winner===name?'<small class="winner-tag">✓ WINNER</small>':''}</div>`;
 const gameRows=(games.length?games:[{game:1,status:'scheduled'},{game:2,status:'scheduled'}]).map(g=>{
  const index=videoIndex(stage,pair,g.game),a=resultFor(g,pair.a)?.totalScore,b=resultFor(g,pair.b)?.totalScore;
  const score=g.status==='complete'?`<span class="game-score"><span class="${a>b?'score-leading':''}">${a.toLocaleString('en-US')}</span> : <span class="${b>a?'score-leading':''}">${b.toLocaleString('en-US')}</span></span>`:`<span class="game-pending">${esc(g.status)}</span>`;
  const replay=index>=0?`<button class="replay-mini" data-replay="${index}" aria-label="Watch ${esc(pair.a)} versus ${esc(pair.b)}, game ${g.game}" aria-pressed="${index===selectedReplay}">▶ Replay</button>`:`<small class="recording-note">${g.status==='complete'?((D.replayPolicy?.archivedStages?.includes(stage)||D.replayPolicy?.archivedGames?.some(v=>v.stage===stage&&v.matchA===pair.a&&v.matchB===pair.b&&v.game===g.game))?'Video archived · recording retained':stage.startsWith('playoff-')?'Recording retained':'No video'):''}</small>`;
  return `<div class="result-game-row"><span>GAME ${g.game}</span>${score}${replay}</div>`;
 }).join('');
 const resolution=pair.resolution&&complete?`<p class="resolution-note">Decided by ${esc({'combined-score':'combined official score','round-first-places':'round first places','extra-game-score':'extra game score','registered-draw-of-lots':'registered draw of lots'}[pair.resolution]||pair.resolution)}.</p>`:'';
 const reviewNotes=(D.resultNotes||[]).filter(n=>n.stage===stage&&n.matchId===pair.matchId).map(n=>`<a class="score-review" href="${esc(n.url)}" target="_blank" rel="nofollow noopener" title="${esc(n.explanation)}">${n.kind==='void'?'Game '+n.game+' replaced after replay corruption ⓘ':'Game '+n.game+' score verified after review ⓘ'}</a>`).join('');
 return `<article class="match-card ${complete?'decided':''}"><div class="match-kicker"><span>${esc(pair.matchId||stageLabel(stage))} / TWO × 20 ROUNDS</span><span>${complete?(winner?'FINAL':'DRAW'):done?done+'/2 GAMES VERIFIED':esc(pair.status)}</span></div><div class="pair-top">${entrant(pair.a,0)}<span class="versus">VS</span>${entrant(pair.b,1)}</div><div class="result-games">${gameRows}</div>${resolution}${reviewNotes}</article>`;
}
function swissTable(limit){return '<table><thead><tr><th>Seed</th><th>Bot</th><th>Pts</th><th>Buchholz</th><th>Share</th></tr></thead><tbody>'+(D.standings||[]).slice(0,limit).map((r,i)=>`<tr class="${i<8?'qualified-row':''}"><td>${i+1}</td><td>${botName(r.name)}${i<8?'<small class="seed-tag">TOP 8</small>':''}</td><td>${r.points}</td><td>${r.buchholz}</td><td>${(r.scoreShare*100).toFixed(1)}%</td></tr>`).join('')+'</tbody></table>'}
function livesTable(){return '<table><thead><tr><th>Seed</th><th>Bot</th><th>Wins</th><th>Losses</th><th>Status</th></tr></thead><tbody>'+(D.playoffStandings||[]).map(r=>`<tr class="${r.eliminated?'eliminated-row':''}"><td>${r.seed}</td><td>${botName(r.name)}</td><td>${r.wins}</td><td>${r.losses} / 2</td><td>${r.name===D.champion?'🏆 Champion':r.eliminated?'Eliminated':r.losses===1?'Lower bracket':'Upper bracket'}</td></tr>`).join('')+'</tbody></table>'}

function meleeTable(rows){return '<table class="melee-score-table"><thead><tr><th aria-label="Rank">#</th><th>Bot</th><th>Score</th><th>1st</th></tr></thead><tbody>'+rows.map(r=>'<tr'+(r.rank===1?' class="melee-leading"':'')+'><td>'+r.rank+'</td><td>'+botName(r.name,false)+'</td><td>'+Number(r.totalScore).toLocaleString('en-US')+'</td><td>'+r.firstPlaces+'</td></tr>').join('')+'</tbody></table>'}
function addExhibitionLink(){if(!D.exhibition)return;if(page==='home'){const link=document.createElement('a');link.className='feature-match exhibition-feature';link.href='exhibition.html';link.innerHTML='<small>POST-TOURNAMENT EXHIBITION</small><strong>Top-four melee</strong><span>'+D.exhibition.completedGames+' / 2 games verified · Four bots together ↗</span>';$('#featured').append(link)}if(page==='matches'){const link=document.createElement('a');link.href='exhibition.html';link.textContent='Exhibition';$('#stage-tabs').append(link)}}
function renderExhibition(){
 activeStage='exhibition';const e=D.exhibition;
 $('#stage-tabs').innerHTML='<a href="matches.html?stage=playoff-6">Championship</a><a href="exhibition.html" aria-current="page">Exhibition</a>';
 $('#stage-title').textContent='Top-four melee';$('#stage-progress').textContent=e?e.completedGames+' / 2 GAMES VERIFIED':'NOT STARTED';
 $('#stage-note').textContent='Two twenty-round games with all four bots together. Scores belong only to this exhibition; KeelHelm remains tournament champion.';
 $('#matchlist').after($('#replay-room'));
 $('#matchlist').innerHTML=e?e.games.map(g=>{const i=replayItems().findIndex(v=>v.stage==='exhibition'&&v.game===g.game);const button=i>=0?'<button class="replay-mini" data-replay="'+i+'" aria-label="Watch top-four melee, game '+g.game+'">▶ Replay</button>':'<small>Recording retained</small>';return '<article class="match-card"><div class="match-kicker"><span>GAME '+g.game+' / 20 ROUNDS</span>'+button+'</div>'+(g.status==='complete'?meleeTable(g.official.results):'<p class="melee-pending">'+esc(g.status)+'</p>')+'</article>'}).join(''):'<p class="notice">No exhibition games have been published yet.</p>';
 $('#melee-standings').innerHTML=e?meleeTable(e.standings)+'<p class="profile-results-note">Combined official score, then round first places. Exact ties share rank. No championship points or eliminations.</p>':'';
 renderVideo();positionReplayRoom();
}

function renderHome(){
 const latest=latestStage(),all=stages();$('#homegames').textContent=all.reduce((n,r)=>n+r.completedGames,0);
 $('#featured').innerHTML=latest.draw.map(p=>{const score=sumScores(p);return `<a class="feature-match" href="matches.html?stage=${latest.stage}"><small>${esc(p.matchId||stageLabel(latest.stage))} / ${esc(p.status)}</small><strong>${botName(p.a,false)} <span style="color:#6987a7;font-size:10px">vs</span> ${botName(p.b,false)}</strong><div class="feature-score">${p.games?.some(g=>g.status==='complete')?score[0].toLocaleString('en-US')+' : '+score[1].toLocaleString('en-US'):'— : —'}</div><span class="feature-state">${p.winner?botName(p.winner,false)+' wins ↗':'View the matchup ↗'}</span></a>`}).join('');
 $('#homestandings').innerHTML=swissTable(8);$('#homelives').innerHTML=`<span class="pill">${D.champion?'CHAMPION: '+esc(D.champion):D.playoffStandings?.length?D.playoffStandings.filter(r=>!r.eliminated).length+' BOTS STILL IN CONTENTION':'PLAYOFF FIELD CONFIRMED'}</span>`;
}
function positionReplayRoom(){const room=$('#replay-room');if(!room)return;const button=document.querySelector('[data-replay="'+selectedReplay+'"]');if(window.matchMedia('(max-width:900px)').matches&&button){button.closest('.match-card').after(room);room.classList.add('inline-replay-room')}else{$('#matchlist').after(room);room.classList.remove('inline-replay-room')}}
function renderMatches(){
 const all=stages();if(!all.some(r=>r.stage===activeStage))activeStage=latestStage().stage;
 const current=all.find(r=>r.stage===activeStage);
 $('#stage-tabs').innerHTML=all.map(r=>`<a href="matches.html?stage=${r.stage}" ${r.stage===activeStage?'aria-current="page"':''}>${esc(stageLabel(r.stage))}</a>`).join('');
 if(lastRenderedStage!==activeStage){const tab=$('#stage-tabs [aria-current="page"]');if(tab){const bounds=tab.getBoundingClientRect(),parent=$('#stage-tabs'),outer=parent.getBoundingClientRect();parent.scrollLeft+=bounds.left-outer.left-(parent.clientWidth-bounds.width)/2}lastRenderedStage=activeStage}
 $('#stage-title').textContent=stageLabel(current.stage);$('#stage-progress').textContent=current.completedGames+' / '+current.plannedGames+' GAMES VERIFIED';
 $('#stage-note').textContent=current.stage.startsWith('playoff-')?(current.status==='incomplete'?'This wave is paused for validation review. Validated scores are retained; incomplete pairings do not advance.':current.status==='complete'?(current.stage==='playoff-1'?'Opening playoff round complete. Winners stay in the upper bracket; first-time losers continue in the lower bracket.':'This playoff wave is complete. The updated lives table records advancement and elimination.'):'Two games per pairing. One loss opens the lower-bracket path; only two playoff losses eliminate a bot.'):'Two 20-round games per pairing. The larger combined official score wins.';
 if((D.resultNotes||[]).some(n=>n.stage===current.stage&&n.kind==='void'))$('#stage-note').textContent+=' The original corrupt-recording game was voided; only its replacement counts.';
 if((D.resultNotes||[]).some(n=>n.stage===current.stage&&n.kind!=='void'))$('#stage-note').textContent+=' One original engine result was retained after a documented validation review.';
 $('#matchlist').after($('#replay-room'));
 $('#matchlist').innerHTML=current.draw.map(p=>matchCard(p,current.stage)).join('');
 $('#standingbox').innerHTML=swissTable();$('#lives-section').hidden=!D.playoffStandings?.length;$('#playofflives').innerHTML=livesTable();
 renderVideo();positionReplayRoom();
}
function guestText(text){let value=esc(text);for(const bot of D.profiles.filter(b=>b.kind==='Guest'))value=value.split(esc(bot.name)).join('<span class="guest-name">'+esc(bot.name)+'</span>');return value}
function updatePlayAll(){const button=$('#playall');button.textContent=playAllReplays?'Stop after this game':'Play all this round';button.setAttribute('aria-label','Play all '+stageLabel(activeStage)+' replays');button.setAttribute('aria-pressed',String(playAllReplays))}
function downloadReplay(url,signal,onProgress){
 if(typeof XMLHttpRequest==='undefined')return fetch(url,{signal}).then(r=>{if(!r.ok)throw Error('HTTP '+r.status);return r.blob()});
 return new Promise((resolve,reject)=>{
  const xhr=new XMLHttpRequest(),abort=()=>xhr.abort(),cleanup=()=>signal.removeEventListener('abort',abort);
  if(signal.aborted){const error=Error('Download cancelled');error.name='AbortError';reject(error);return}
  xhr.open('GET',url,true);xhr.responseType='blob';xhr.timeout=180000;
  xhr.onprogress=e=>onProgress(e.loaded,e.lengthComputable?e.total:0);
  xhr.onload=()=>{cleanup();if(xhr.status>=200&&xhr.status<300)resolve(xhr.response);else reject(Error('HTTP '+xhr.status))};
  xhr.onerror=()=>{cleanup();reject(Error('Video transfer failed'))};xhr.ontimeout=()=>{cleanup();reject(Error('Video transfer timed out'))};
  xhr.onabort=()=>{cleanup();const error=Error('Download cancelled');error.name='AbortError';reject(error)};
  signal.addEventListener('abort',abort,{once:true});xhr.send();
 });
}
async function downloadReplayMedia(item,signal,onProgress){
 if(!item.parts?.length)return downloadReplay(item.mp4,signal,onProgress);
 const pieces=[];let completed=0;
 for(const part of item.parts){
  const blob=await downloadReplay(part.url,signal,loaded=>onProgress(completed+loaded,item.sizeBytes));
  if(blob.size!==part.sizeBytes)throw Error('Video part size differs');
  if(part.sha256&&globalThis.crypto?.subtle){const digest=await crypto.subtle.digest('SHA-256',await blob.arrayBuffer());const actual=[...new Uint8Array(digest)].map(b=>b.toString(16).padStart(2,'0')).join('');if(actual!==part.sha256)throw Error('Video part integrity check failed')}
  if(signal.aborted){const error=Error('Download cancelled');error.name='AbortError';throw error}
  pieces.push(blob);completed+=blob.size;
 }
 if(completed!==item.sizeBytes)throw Error('Video length differs');
 return new Blob(pieces,{type:'video/mp4'});
}
function configureReplayDownload(item,v){const link=$('#videodownload');link.download=item.fileName||(item.mp4||'replay.mp4').split('/').pop();const url=v.dataset.objectUrl||item.mp4;if(url){link.href=url;link.textContent='Download this MP4 ↗';link.onclick=null}else{link.href='#replay';link.textContent='Load replay to enable download';link.onclick=e=>{e.preventDefault();loadReplay()}}}
async function loadReplay(){
 const item=replayItems()[selectedReplay],v=$('#replayvideo'),button=$('#loadreplay');if(!item||!v)return;
 const requestId=++replayLoadId;
 try{
  if(!v.dataset.objectUrl){replayAbort?.abort();replayAbort=new AbortController();button.disabled=true;button.textContent='Loading replay…';const blob=await downloadReplayMedia(item,replayAbort.signal,(loaded,total)=>{if(requestId===replayLoadId)button.textContent='Loading replay · '+Math.min(100,Math.round(loaded/(total||item.sizeBytes)*100))+'%'});if(requestId!==replayLoadId)return;v.dataset.objectUrl=URL.createObjectURL(blob);v.src=v.dataset.objectUrl;v.load();configureReplayDownload(item,v)}
  if(requestId!==replayLoadId)return;v.playbackRate=Number($('#playbackspeed').value);button.disabled=false;button.textContent='Play replay';
  try{await v.play();if(requestId===replayLoadId)button.hidden=true}catch(e){if(requestId===replayLoadId){button.hidden=false;button.textContent='Tap to play replay'}}
 }catch(e){if(requestId===replayLoadId&&e.name!=='AbortError'){button.disabled=false;button.hidden=false;button.textContent='Retry video load';button.dataset.loadError=e.message;console.warn('Replay download failed',e.name,e.message)}}
}
function chooseReplay(index,autoplay=false){
 const list=replayItems();if(!Number.isInteger(index)||index<0||index>=list.length)return;
 selectedReplay=index;renderVideo();positionReplayRoom();document.querySelectorAll('[data-replay]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.replay)===index)));
 if(autoplay){loadReplay();if(window.matchMedia('(max-width:900px)').matches)$('#replay-room').scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion:reduce)').matches?'auto':'smooth',block:'start'})}
}
function renderVideo(){
 const list=replayItems(),item=selectedReplay===null?null:list[selectedReplay];$('#replay').hidden=!item;$('#watch-placeholder').hidden=!!item;if(!item)return;
 const v=$('#replayvideo'),source=$('#replaysource'),button=$('#loadreplay');
 const sourceKey=item.sha256||item.mp4;if(v.dataset.source!==sourceKey){++replayLoadId;replayAbort?.abort();v.pause();if(v.dataset.objectUrl)URL.revokeObjectURL(v.dataset.objectUrl);v.removeAttribute('src');source.removeAttribute('src');v.dataset.objectUrl='';v.dataset.source=sourceKey;v.poster=item.poster;$('#replaychapters').src=item.chaptersFile;button.hidden=false;button.disabled=false;button.dataset.loadError='';button.textContent='Play replay · '+Math.ceil(item.sizeBytes/1000000)+' MB';v.load()}
 button.onclick=loadReplay;v.onplaying=()=>{button.hidden=true};$('#playbackspeed').onchange=e=>{v.playbackRate=Number(e.target.value)};
 $('#videotitle').innerHTML=guestText(item.title);$('#videolabel').textContent=item.label;configureReplayDownload(item,v);
 $('#roundjump').innerHTML='<option value="">Battle round…</option>'+item.chapters.map(c=>`<option value="${c.startSeconds}">Round ${String(c.round).padStart(2,'0')}</option>`).join('');$('#roundjump').disabled=v.readyState<1;
 v.onloadedmetadata=()=>{$('#roundjump').disabled=false;v.playbackRate=Number($('#playbackspeed').value)};
 $('#roundjump').onchange=e=>{if(e.target.value==='')return;v.currentTime=Number(e.target.value);e.target.value=''};
 const scope=replayScope(),position=scope.indexOf(selectedReplay);
 $('#replayprevious').disabled=position<=0;$('#replaynext').disabled=position<0||position>=scope.length-1;
 $('#replayprevious').onclick=()=>{playAllReplays=false;chooseReplay(scope[position-1],true)};$('#replaynext').onclick=()=>{playAllReplays=false;chooseReplay(scope[position+1],true)};
 updatePlayAll();$('#playall').hidden=!scope.length;$('#playall').onclick=()=>{playAllReplays=!playAllReplays;if(playAllReplays)chooseReplay(scope[0],true);else updatePlayAll()};
 v.onended=()=>{if(!playAllReplays)return;const currentScope=replayScope(),currentPosition=currentScope.indexOf(selectedReplay);if(currentPosition>=0&&currentPosition+1<currentScope.length)chooseReplay(currentScope[currentPosition+1],true);else{playAllReplays=false;updatePlayAll();$('#replaystatus').textContent='All replays in this round finished.'}};
 $('#replaystatus').textContent='Replay '+(position+1)+' of '+scope.length+' · '+stageLabel(activeStage)+(playAllReplays?' · Playing all in order':'');
}
function render(){
 $('#status').textContent=D.status.toUpperCase();$('#updated').textContent='Snapshot: '+D.updated;
 if(page==='home')renderHome();if(page==='matches')renderMatches();if(page==='exhibition')renderExhibition();addExhibitionLink();if(page==='bots')roster();
 if($('#profile').open&&profileId)profile(profileId);
}
function loadData(){fetch('data.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error(r.status);return r.json()}).then(d=>{if(!D||D.updated!==d.updated){D=d;render()}}).catch(()=>{if(!D){const banner=document.createElement('p');banner.className='notice';banner.textContent='Tournament results could not load. Please refresh to try again.';$('main').prepend(banner)}})}
if(page==='home'&&location.hash){const target={'#replay':'matches.html?stage=swiss-4#replay-room','#playoffs':'matches.html','#swissmatches':'matches.html?stage=swiss-4','#practice':'matches.html?stage=practice','#bots':'bots.html','#rules':'rules.html','#standings':'matches.html#standingbox'}[location.hash];if(target)location.replace(target)}
document.addEventListener('click',e=>{const replay=e.target.closest('[data-replay]');if(replay){playAllReplays=false;chooseReplay(Number(replay.dataset.replay),true)}});
document.querySelectorAll('.filter').forEach(b=>b.onclick=()=>{kind=b.dataset.filter;document.querySelectorAll('.filter').forEach(x=>x.classList.toggle('active',x===b));roster()});if($('#search'))$('#search').oninput=e=>{query=e.target.value.toLowerCase();roster()};
$('#profile').onclose=()=>{profileId=null};$('.close').onclick=()=>$('#profile').close();$('#profile').onclick=e=>{if(e.target===$('#profile'))$('#profile').close()};
window.matchMedia('(max-width:900px)').addEventListener('change',()=>{if(page==='matches'||page==='exhibition')positionReplayRoom()});
loadData();setInterval(()=>{if(!document.hidden)loadData()},15000);
