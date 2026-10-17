"""Do not retain a missing frontend asset across local rebuilds."""
from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles


class NoNegativeCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        try:
            response = await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code >= 400:
                exc.headers = {**(exc.headers or {}), 'Cache-Control': 'no-store'}
            raise
        if response.status_code >= 400:
            response.headers['Cache-Control'] = 'no-store'
        return response
