"""Compress display JSON; serve build-time gzip without spending request CPU."""
import mimetypes
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.middleware.gzip import GZipMiddleware
from starlette.staticfiles import StaticFiles


class DisplayCompression:
    prefixes = ('/api/leaderboards', '/api/human/leaderboards', '/api/human/players/',
                '/api/minigame-rankings/games/', '/api/minigame-rankings/overall',
                '/api/minigame-rankings/catalog', '/api/minigame-rankings/me/records',
                '/api/gamer/replays/')

    def __init__(self, app):
        self.app = app
        self.compressed = GZipMiddleware(app, minimum_size=1000, compresslevel=4)

    async def __call__(self, scope, receive, send):
        target = self.compressed if scope['type'] == 'http' and scope['path'].startswith(self.prefixes) else self.app
        await target(scope, receive, send)


def accepts_gzip(value):
    qualities = {}
    for part in value.lower().split(','):
        coding, *params = part.strip().split(';')
        try:
            qualities[coding] = next((float(p.strip()[2:]) for p in params if p.strip().startswith('q=')), 1.0)
        except ValueError:
            qualities[coding] = 0
    return qualities.get('gzip', qualities.get('*', 0)) > 0


class CacheControlledStaticFiles(StaticFiles):
    def __init__(self, *args, cache_control='', **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_control = cache_control

    async def get_response(self, path, scope):
        headers = Headers(scope=scope)
        response = None
        # Range semantics remain those of the original representation.
        if accepts_gzip(headers.get('accept-encoding', '')) and 'range' not in headers:
            candidate = path.rstrip('/') + '/index.html' if self.html and (path == '.' or scope['path'].endswith('/')) else path
            if candidate.startswith('./'):
                candidate = candidate[2:]
            try:
                response = await super().get_response(candidate + '.gz', scope)
                if response.status_code not in (200, 304):
                    response = None
                else:
                    response.headers['Content-Encoding'] = 'gzip'
                    response.headers['Content-Type'] = mimetypes.guess_type(candidate)[0] or 'application/octet-stream'
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
        if response is None:
            response = await super().get_response(path, scope)
        response.headers.add_vary_header('Accept-Encoding')
        if response.status_code in (200, 304):
            response.headers['Cache-Control'] = self.cache_control or (
                'public, max-age=31536000, immutable' if path.replace('\\', '/').startswith(('assets/', 'wasm/')) else 'no-cache')
        return response
