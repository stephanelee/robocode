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
                async for message in ws:
                    await ws.send(message);await ws.send('{"type":"GameEndedEventForBot"}')
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
                    await front.send(intent);self.assertEqual(intent,await front.recv());self.assertEqual('GameEndedEventForBot',json.loads(await front.recv())['type'])
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
    async def probe_case(self,role=None,backend_abort=False):
        async def backend(ws):
            await ws.send('{"type":"ServerHandshake","sessionId":"ok"}')
            if backend_abort:
                await asyncio.sleep(.03);ws.transport.abort();return
            try:
                async for raw in ws:pass
            except Exception:pass
        native=await serve(backend,'127.0.0.1',0);relay=Relay(f'ws://127.0.0.1:{native.sockets[0].getsockname()[1]}',{'Alpha':'1'},None)
        proxy=await serve(relay.handler,'127.0.0.1',0)
        path='/identity/Alpha'if role=='BotHandshake'else'/'
        try:
            async with connect(f'ws://127.0.0.1:{proxy.sockets[0].getsockname()[1]}'+path)as front:
                await front.recv()
                if role:
                    msg=dict(type=role,sessionId='ok')
                    if role=='BotHandshake':msg.update(name='Alpha',version='1')
                    await front.send(json.dumps(msg));await asyncio.sleep(.03)
                if not backend_abort:front.transport.abort()
                await asyncio.sleep(.08)
            if role is None and not backend_abort:
                self.assertFalse(relay.errors);self.assertTrue(relay.rows[0]['discoveryProbe']);self.assertEqual(0,relay.rows[0].get('upInputCount',0))
            else:self.assertTrue(relay.errors);self.assertFalse(relay.rows[0].get('discoveryProbe',False))
        finally:
            relay.closing=True;proxy.close();await asyncio.wait_for(proxy.wait_closed(),3);await relay.close();native.close();await asyncio.wait_for(native.wait_closed(),3)
        self.assertFalse(native.connections)
    async def test_empty_root_readiness_abort_is_not_a_gameplay_failure(self):await self.probe_case()
    async def test_bot_and_controller_abrupt_loss_remain_failures(self):
        for role in ['BotHandshake','ControllerHandshake']:
            with self.subTest(role=role):await self.probe_case(role)
    async def test_backend_loss_is_never_a_front_probe(self):await self.probe_case(backend_abort=True)
    async def normal_close_case(self,role,server_closes=False,terminal=False):
        async def backend(ws):
            await ws.send('{"type":"ServerHandshake","sessionId":"ok"}')
            try:
                await ws.recv()
                if terminal:await ws.send(json.dumps(dict(type='GameEndedEventForBot'if role=='BotHandshake'else'GameEndedEventForObserver')))
                if server_closes:await ws.close();return
                await ws.wait_closed()
            except Exception:pass
        native=await serve(backend,'127.0.0.1',0);relay=Relay(f'ws://127.0.0.1:{native.sockets[0].getsockname()[1]}',{'Alpha':'1'},None);proxy=await serve(relay.handler,'127.0.0.1',0)
        path='/identity/Alpha'if role=='BotHandshake'else'/'
        try:
            async with connect(f'ws://127.0.0.1:{proxy.sockets[0].getsockname()[1]}'+path)as front:
                await front.recv();msg=dict(type=role,sessionId='ok')
                if role=='BotHandshake':msg.update(name='Alpha',version='1')
                await front.send(json.dumps(msg));await asyncio.sleep(.03)
                if terminal:await front.recv()
                if not server_closes:await front.close()
                await asyncio.sleep(.06)
            if terminal:
                self.assertFalse(relay.errors);self.assertEqual('game-ended',relay.rows[0]['expectedClose'])
            else:self.assertTrue(relay.errors)
        finally:
            relay.closing=True;proxy.close();await asyncio.wait_for(proxy.wait_closed(),3);await relay.close();native.close();await asyncio.wait_for(native.wait_closed(),3)
        self.assertFalse(native.connections)
    async def test_premature_normal_bot_controller_and_backend_close_are_fatal(self):
        for role,backend in [('BotHandshake',False),('ControllerHandshake',False),('BotHandshake',True)]:
            with self.subTest(role=role,backend=backend):await self.normal_close_case(role,backend)
    async def test_forwarded_bot_or_observer_terminal_makes_close_expected(self):
        for role in ['BotHandshake','ObserverHandshake']:
            with self.subTest(role=role):await self.normal_close_case(role,terminal=True)
    async def test_authoritative_observer_end_allows_bot_shutdown_without_own_end_packet(self):
        async def backend(ws):
            await ws.send('{"type":"ServerHandshake","sessionId":"ok"}')
            try:
                role=json.loads(await ws.recv())['type']
                if role=='ObserverHandshake':await ws.send('{"type":"GameEndedEventForObserver"}')
                await ws.wait_closed()
            except Exception:pass
        native=await serve(backend,'127.0.0.1',0);relay=Relay(f'ws://127.0.0.1:{native.sockets[0].getsockname()[1]}',{'Alpha':'1'},None);proxy=await serve(relay.handler,'127.0.0.1',0);base=f'ws://127.0.0.1:{proxy.sockets[0].getsockname()[1]}'
        try:
            async with connect(base+'/identity/Alpha')as bot:
                await bot.recv();await bot.send('{"type":"BotHandshake","name":"Alpha","version":"1","sessionId":"ok"}');await asyncio.sleep(.02)
                async with connect(base)as observer:
                    await observer.recv();await observer.send('{"type":"ObserverHandshake","sessionId":"ok"}');await observer.recv();await asyncio.sleep(.02)
                    await bot.close();await observer.close()
            await asyncio.sleep(.05);self.assertFalse(relay.errors);self.assertTrue(relay.game_ended)
            self.assertTrue(all(r.get('expectedClose')=='game-ended'for r in relay.rows))
        finally:
            relay.closing=True;proxy.close();await asyncio.wait_for(proxy.wait_closed(),3);await relay.close();native.close();await asyncio.wait_for(native.wait_closed(),3)
        self.assertFalse(native.connections)
if __name__=='__main__':unittest.main()
