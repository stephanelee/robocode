import asyncio,json,unittest
from types import SimpleNamespace
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from paired_transport import Relay,rewrite_handshake

class RewriteContracts(unittest.TestCase):
    def test_only_registered_position_changes_and_session_must_be_the_actual_one(self):
        raw=' {"type":"BotHandshake", "name":"Alpha", "version":"1", "sessionId":"aa","secret":"fixture"}'
        self.assertEqual(raw,rewrite_handshake(raw,'Alpha','1',None,'aa'))
        after=json.loads(rewrite_handshake(raw,'Alpha','1',[200,300,45],'aa'));before=json.loads(raw)
        self.assertEqual(dict(x=200,y=300,direction=45),after.pop('initialPosition'));self.assertEqual(before,after)
        for name,version,session in [('Other','1','aa'),('Alpha','2','aa'),('Alpha','1','zz')]:
            with self.assertRaises(AssertionError):rewrite_handshake(raw,name,version,[1,2,3],session)

class SocketContracts(unittest.IsolatedAsyncioTestCase):
    async def exercise(self,fixed):
        counter=0;received=[];backends=[]
        async def backend(ws):
            nonlocal counter
            session=['zz','aa'][counter];counter+=1;backends.append(ws)
            await ws.send(json.dumps(dict(type='ServerHandshake',sessionId=session)))
            try:
                received.append(json.loads(await ws.recv()))
                async for message in ws:await ws.send(message)
            except Exception:pass
        native=await serve(backend,'127.0.0.1',0)
        relay=Relay(f'ws://127.0.0.1:{native.sockets[0].getsockname()[1]}',{'Alpha':'1','Beta':'1'},{'Alpha':[200,300,45],'Beta':[600,300,225]}if fixed else None)
        proxy=None
        try:
            await relay.prepare();proxy=await serve(relay.handler,'127.0.0.1',0)
            for name in ['Beta','Alpha']:
                async with connect(f'ws://127.0.0.1:{proxy.sockets[0].getsockname()[1]}/identity/{name}')as front:
                    hello=json.loads(await front.recv());session=hello['sessionId']
                    if fixed:self.assertEqual({'Alpha':'aa','Beta':'zz'}[name],session)
                    raw=json.dumps(dict(type='BotHandshake',name=name,version='1',sessionId=session))
                    await front.send(raw)
                    intent=' { "type" : "BotIntent", "turnNumber":1, "targetSpeed":-3.25 } '
                    await front.send(intent);self.assertEqual(intent,await front.recv())
            await asyncio.sleep(.05)
            self.assertFalse(relay.errors);self.assertEqual({'Alpha','Beta'},relay.seen);self.assertFalse(relay.pool)
            self.assertEqual(2,len(received))
            for row in relay.rows:
                self.assertEqual(row['downInput'],row['downOutput'])
                self.assertEqual(row['upAfterHandshakeInput'],row['upAfterHandshakeOutput'])
                self.assertEqual(int(fixed),row['rewrites'])
                if not fixed:self.assertEqual(row['upInput'],row['upOutput'])
        finally:
            relay.closing=True
            if proxy:proxy.close();await asyncio.wait_for(proxy.wait_closed(),3)
            await relay.close();native.close();await asyncio.wait_for(native.wait_closed(),3)
        self.assertFalse(native.connections)
    async def test_fixed_names_receive_their_own_sorted_real_backend_sessions(self):await self.exercise(True)
    async def test_random_mode_is_byte_preserving(self):await self.exercise(False)
    async def test_wrong_identity_is_reported_and_backend_is_closed(self):
        async def backend(ws):
            await ws.send('{"type":"ServerHandshake","sessionId":"ok"}')
            await ws.wait_closed()
        native=await serve(backend,'127.0.0.1',0);relay=Relay(f'ws://127.0.0.1:{native.sockets[0].getsockname()[1]}',{'Alpha':'1'},None)
        proxy=await serve(relay.handler,'127.0.0.1',0)
        try:
            async with connect(f'ws://127.0.0.1:{proxy.sockets[0].getsockname()[1]}/identity/Alpha')as f:
                await f.recv();await f.send('{"type":"BotHandshake","name":"Wrong","version":"1","sessionId":"ok"}')
                await asyncio.sleep(.05)
            self.assertTrue(relay.errors)
        finally:
            relay.closing=True;proxy.close();await asyncio.wait_for(proxy.wait_closed(),3);await relay.close();native.close();await asyncio.wait_for(native.wait_closed(),3)
        self.assertFalse(native.connections)
if __name__=='__main__':unittest.main()
