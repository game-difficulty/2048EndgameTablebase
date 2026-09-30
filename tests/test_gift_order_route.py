from unittest.mock import patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.live import routes


def test_gift_order_routes_use_authenticated_account_and_preserve_origin_checks():
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client, patch.object(routes, 'resolve_hub', return_value=routes.hub), \
            patch.object(routes, 'require_user', return_value={'id': 7}), \
            patch.object(routes.gifts, 'set_gift_order', return_value={'gift_order': ['two']}) as save:
        for path in ('/api/live/gifts/preferences/order',
                     '/api/live/rooms/ai-classic/gifts/preferences/order'):
            response = client.post(path, json={'gift_ids': ['two'], 'user_id': 99})
            assert response.status_code == 200
            save.assert_called_with(7, ['two'])
        save.reset_mock()
        with patch.object(routes, 'same_origin', side_effect=HTTPException(403, 'cross_origin')):
            assert client.post(path, json={'gift_ids': ['two']}).status_code == 403
        save.assert_not_called()
        with patch.object(routes, 'require_user', side_effect=HTTPException(401)):
            assert client.post(path, json={'gift_ids': ['two']}).status_code == 401
        save.assert_not_called()
