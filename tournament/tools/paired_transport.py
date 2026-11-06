"""Local official-protocol transport; unchanged engine/bots, no policy state interception."""
import asyncio,hashlib,json,os,socket,subprocess,sys
from pathlib import Path
from urllib.parse import unquote
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosedOK,ConnectionClosedError

SERVER_SHA='16c277795bd823c9eda0b650f1d09a418ff35cfe43d59b820c52d5e6816f905c'
def rewrite_handshake(raw,name,version,position,session):
    value=json.loads(raw)
    assert value['type']=='BotHandshake'and value['name']==name and value['version']==version
    assert value['sessionId']==session,'A real connection must echo its own issued session ID'
    if position is None:return raw
    before={k:v for k,v in value.items()if k!='initialPosition'}
    value['initialPosition']=dict(zip(('x','y','direction'),position))
    assert before=={k:v for k,v in value.items()if k!='initialPosition'}
    return json.dumps(value,separators=(',',':'),ensure_ascii=False)

def is_empty_front_probe(row):
    hello=row.get('serverHandshake','');data=hello.encode();digest=hashlib.sha256(len(data).to_bytes(8,'big')+data).hexdigest()
    try:valid=json.loads(hello).get('type')=='ServerHandshake'
    except (ValueError,AttributeError):valid=False
    return (valid and digest==row.get('downOutput'))and (row['path']=='/' and 'name'not in row and 'roleType'not in row
            and row.get('upInputCount',0)==row.get('upOutputCount',0)==0
            and row.get('downInputCount',0)==row.get('downOutputCount',0)==1
            and row.get('downInput')==row.get('downOutput')
            and row.get('serverHandshakeSent')is True and row.get('closeOrigin')=='up')

class Relay:
    def __init__(self,url,identities,positions):
        self.url=url;self.identities=identities;self.positions=positions;self.pool={};self.seen=set();self.errors=[];self.rows=[];self.closing=False
    async def open_backend(self):
        ws=await connect(self.url,max_size=None,ping_interval=None,open_timeout=10)
        try:
            raw=await asyncio.wait_for(ws.recv(),10);msg=json.loads(raw)
            assert msg['type']=='ServerHandshake'and isinstance(msg['sessionId'],str)
            return ws,raw,msg['sessionId']
        except BaseException:
            await ws.close();raise
    async def prepare(self):
        if self.positions is None:return
        rows=[]
        try:
            for _ in self.identities:rows.append(await self.open_backend())
            assert len({r[2]for r in rows})==len(rows)
            self.pool=dict(zip(sorted(self.identities),sorted(rows,key=lambda r:r[2])))
        except BaseException:
            for ws,_,_ in rows:await ws.close()
            raise
    async def handler(self,front):
        backend=None;tasks=[];row={'path':front.request.path,'messagesUp':0,'messagesDown':0,'rewrites':0};self.rows.append(row)
        hashes={k:hashlib.sha256()for k in ('upInput','upOutput','downInput','downOutput','upAfterHandshakeInput','upAfterHandshakeOutput')}
        def digest(key,msg):
            data=msg.encode()if isinstance(msg,str)else msg;hashes[key].update(len(data).to_bytes(8,'big')+data);row[key+'Count']=row.get(key+'Count',0)+1
        try:
            path=front.request.path
            name=unquote(path[len('/identity/'):])if path.startswith('/identity/')else None
            assert name is None or name in self.identities
            assert name is not None or path=='/'
            if name is not None:
                assert name not in self.seen,'Duplicate/reconnected bot';self.seen.add(name)
            backend,hello,session=self.pool.pop(name)if self.positions is not None and name is not None else await self.open_backend()
            row['serverHandshake']=hello
            digest('downInput',hello);await front.send(hello);digest('downOutput',hello);row['messagesDown']+=1;row['serverHandshakeSent']=True
            async def up():
                handshaken=False
                async for raw in front:
                    digest('upInput',raw);out=raw
                    if not handshaken:
                        msg=json.loads(raw);row['roleType']=msg.get('type');row['roleHandshake']=raw
                        if name is not None:
                            out=rewrite_handshake(raw,name,self.identities[name],None if self.positions is None else self.positions[name],session)
                            row['handshakeBefore']=raw;row['handshakeAfter']=out;row['name']=name;row['serverSessionId']=session
                            row['rewrites']+=int(out!=raw)
                        else:assert msg['type']in ('ObserverHandshake','ControllerHandshake')
                        handshaken=True
                    else:
                        digest('upAfterHandshakeInput',raw)
                    await backend.send(out);digest('upOutput',out);row['messagesUp']+=1
                    if 'upAfterHandshakeInputCount'in row:digest('upAfterHandshakeOutput',out)
            async def down():
                async for raw in backend:
                    digest('downInput',raw);await front.send(raw);digest('downOutput',raw);row['messagesDown']+=1
            tasks=[asyncio.create_task(up(),name='up'),asyncio.create_task(down(),name='down')]
            done,pending=await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:task.result();row.setdefault('closeOrigin',task.get_name())
                except BaseException:row['closeOrigin']=task.get_name();raise
            for task in pending:task.cancel()
        except ConnectionClosedError as error:
            row.update({k:v.hexdigest()for k,v in hashes.items()})
            if is_empty_front_probe(row) and error.rcvd is None and error.sent is None:
                row['discoveryProbe']=True;row['probeClose']=repr(error)
            elif not self.closing:self.errors.append(repr(error));row['error']=repr(error)
        except ConnectionClosedOK:pass
        except asyncio.CancelledError:raise
        except BaseException as e:
            if not self.closing:self.errors.append(repr(e));row['error']=repr(e)
        finally:
            for task in tasks:
                if not task.done():task.cancel()
            if tasks:await asyncio.gather(*tasks,return_exceptions=True)
            if backend is not None:await backend.close()
            row.update({k:v.hexdigest()for k,v in hashes.items()})
            if is_empty_front_probe(row):row['discoveryProbe']=True
    async def close(self):
        self.closing=True
        for ws,_,_ in self.pool.values():await ws.close()
        self.pool.clear()

def free_port():
    with socket.socket()as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]

async def run(config):
    run=Path(config['run']);java=config['java'];server=None;child=None;relay=None;proxy=None;log=(run/'server.log').open('x');report={}
    try:
        jar=Path(config['serverJar']);assert hashlib.sha256(jar.read_bytes()).hexdigest()==SERVER_SHA
        port=free_port();url=f'ws://127.0.0.1:{port}'
        command=[java,'-jar',str(jar),'-p',str(port),'-i','-t','-1']
        server=await asyncio.create_subprocess_exec(*command,stdout=log,stderr=log)
        for _ in range(200):
            if server.returncode is not None:raise RuntimeError('Server exited during startup')
            try:
                reader,writer=await asyncio.open_connection('127.0.0.1',port);writer.close();await writer.wait_closed();break
            except OSError:await asyncio.sleep(.1)
        else:raise TimeoutError('Server listen timeout')
        relay=Relay(url,config['identities'],config['positions']);await relay.prepare()
        proxy=await serve(relay.handler,'127.0.0.1',0,max_size=None,ping_interval=None)
        proxy_url=f'ws://127.0.0.1:{proxy.sockets[0].getsockname()[1]}'
        cmd=[java,'-Dtournament.serverUrl='+proxy_url,'-Dtournament.recordingDir='+str(run/'recordings'),'-cp',config['runner'],config['adapter'],str(config['rounds']),'900',str(run/'battle.json'),*config['bots']]
        report.update(serverCommand=command,runnerCommand=cmd,serverURL=url,proxyURL=proxy_url)
        child=await asyncio.create_subprocess_exec(*cmd)
        task=asyncio.create_task(child.wait())
        while not task.done():
            if relay.errors:raise RuntimeError('Relay failed: '+str(relay.errors))
            await asyncio.wait([task],timeout=.2)
        assert task.result()==0,'Runner failed'
        assert relay.seen==set(config['identities'])and not relay.pool and not relay.errors
        report['complete']=True
    except BaseException as error:
        report['error']=repr(error);raise
    finally:
        errors=[]
        if relay is not None:relay.closing=True
        for label,process in [('runner',child),('server',server)]:
            if process is None:continue
            try:
                if process.returncode is None:process.terminate()
                try:await asyncio.wait_for(process.wait(),5)
                except asyncio.TimeoutError:
                    process.kill();await asyncio.wait_for(process.wait(),5)
                report[label+'ReturnCode']=process.returncode
            except BaseException as e:errors.append(label+': '+repr(e))
        if proxy is not None:
            try:proxy.close();await asyncio.wait_for(proxy.wait_closed(),10)
            except BaseException as e:errors.append('proxy: '+repr(e))
        if relay is not None:
            try:await asyncio.wait_for(relay.close(),10)
            except BaseException as e:errors.append('relay: '+repr(e))
            report.update(connections=relay.rows,relayErrors=relay.errors)
        report['cleanupErrors']=errors
        if errors:report['complete']=False
        log.close();(run/'transport.json').write_text(json.dumps(report,indent=2)+'\n')
        if errors and 'error'not in report:raise RuntimeError('Transport cleanup failed: '+str(errors))

if __name__=='__main__':asyncio.run(run(json.loads(Path(sys.argv[1]).read_text())))
