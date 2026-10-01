"""Loopback-only manual/browser transport fixture, with disposable databases."""
import asyncio
import contextlib
import os
from pathlib import Path
import tempfile
from fastapi import WebSocket, WebSocketDisconnect
import uvicorn

def create_fixture(port=8789):
    folder=Path(tempfile.mkdtemp(prefix='competition-stream-qa-'))
    os.environ['CLOUD_AUTH_DB']=str(folder/'auth.sqlite3')
    os.environ['COMPETITION_DB']=str(folder/'client.sqlite3')
    os.environ['COMPETITION_ALLOW_DEV_AUTH']='1'
    os.environ['COMPETITION_LIVE_INTERNAL_TOKEN']='loopback-stream-qa'
    os.environ['COMPETITION_LIVE_API_ORIGIN']=f'http://127.0.0.1:{port}'
    from competition.tests.test_client_runtime import setup_game
    service,players=setup_game(folder,'tournament-grand-full-undo-race-3x3')
    from competition.backend.app import create_app
    from backend.live.competition_content import CompetitionMatchContent
    from backend.live.dynamic_rooms import CompetitionMatchRoomProvider
    from fastapi.middleware.cors import CORSMiddleware
    app=create_app()
    app.router.routes=[route for route in app.router.routes if getattr(route,'path','')!='/{path:path}']
    app.add_middleware(CORSMiddleware,allow_origins=['http://127.0.0.1:5198'],allow_methods=['*'],allow_headers=['*'])
    definition=CompetitionMatchRoomProvider._definition(service.list_live_rooms()[0])

    @app.get('/qa/bootstrap')
    async def bootstrap():
        return {'players':[{'user':p.user_id,'room':service.snapshot('MATCH5',p)} for p in [players[0],players[3]]],
                'viewer':players[1].user_id,'public_key':definition.metadata['public_key']}

    @app.websocket('/qa/live')
    async def relay(socket:WebSocket):
        await socket.accept()
        # Constructor's HTTP bootstrap must run off the serving event loop.
        content=await asyncio.to_thread(CompetitionMatchContent,definition)
        queue=asyncio.Queue()
        content.on_update=lambda: queue.put_nowait({'type':'snapshot','match':content.incremental_projection})
        queue.put_nowait({'type':'snapshot','match':content.projection})
        await content.start()
        async def send_updates():
            while True:
                await socket.send_json(await queue.get())
        writer=asyncio.create_task(send_updates())
        try:
            while True:
                await socket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            writer.cancel()
            with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                await writer
            await content.stop()
    return app

if __name__=='__main__':
    uvicorn.run(create_fixture(),host='127.0.0.1',port=8789,log_level='warning')
