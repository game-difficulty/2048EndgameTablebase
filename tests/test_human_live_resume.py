import unittest
from unittest.mock import patch
from types import SimpleNamespace
from backend.live.human_content import HumanPlayContent, HumanStreamRun
from backend.human_play import engine
from backend.gamer_ranked.prng import Xoshiro128StarStar

SEED='00000001000000020000000300000004'

def events(count):
    state=engine.initial('run','4x4',SEED);raw=b''
    for _ in range(count):
        for direction in range(4):
            board,_=engine.move(state['board'],4,4,direction)
            if board!=state['board']:break
        rng=Xoshiro128StarStar(list(state['rng']));index,value=engine.spawn(board,rng)
        step=engine.EVENT.pack(direction|(index<<2)|(64 if value==4 else 0),123)
        raw+=step;state=engine.advance(state,'4x4',step)
    return raw

class Socket:
    def __init__(self,raw):self.raw=raw;self.sent=[]
    async def receive_bytes(self):return b'HLP1\0'+self.raw
    async def send_json(self,value):self.sent.append(value)

class ResumeTests(unittest.IsolatedAsyncioTestCase):
    async def install(self,retained,raw,seq):
        content=HumanPlayContent(SimpleNamespace(id='room',metadata={}))
        content.run=retained;content.verified_milestones={32768}
        hub=SimpleNamespace(broadcast=lambda _:None,snapshot=lambda:{})
        socket=Socket(raw)
        lease={'run':'run','variant':'4x4','seed':SEED,'generation':1}
        with patch('backend.live.human_content.human_rooms.verify_lease',return_value=lease),patch('backend.live.human_content.human_rooms.publisher_seen',return_value=True):
            await content._install_run(socket,hub,{'lease':'signed','seq':seq,'resume':True},{'id':1})
        return content,socket

    async def test_retained_run_only_consumes_tail_and_keeps_milestones(self):
        raw=events(5)
        retained=HumanStreamRun(run_id='run',variant='4x4',seed=SEED,started_at=1,source='Player',raw=raw[:10])
        content,socket=await self.install(retained,raw[10:],5)
        self.assertEqual(socket.sent[0]['start'],2)
        self.assertEqual(content.run.state,engine.advance(engine.initial('run','4x4',SEED),'4x4',raw))
        self.assertEqual(retained.seq,2)
        self.assertEqual(content.verified_milestones,{32768})

    async def test_missing_or_different_run_requests_full_prefix(self):
        raw=events(3)
        old=HumanStreamRun(run_id='old',variant='4x4',seed=SEED,started_at=1,source='Player',raw=b'')
        for retained in [None,old]:
            content,socket=await self.install(retained,raw,3)
            self.assertEqual(socket.sent[0]['start'],0);self.assertEqual(content.run.seq,3)
            self.assertEqual(content.verified_milestones,set())

    async def test_invalid_tail_does_not_mutate_retained_state(self):
        raw=events(2)
        retained=HumanStreamRun(run_id='run',variant='4x4',seed=SEED,started_at=1,source='Player',raw=raw)
        before=retained.snapshot()
        with self.assertRaises(ValueError):await self.install(retained,b'\xff\0\0\0\0',3)
        self.assertEqual(retained.snapshot(),before)
