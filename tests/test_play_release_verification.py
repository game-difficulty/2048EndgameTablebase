import hashlib
import json
from unittest.mock import patch

import pytest

from tools.verify_play_release import verify_files, verify_routes


def test_release_verification_detects_stale_backend(tmp_path):
    source = tmp_path / 'backend.py'
    source.write_bytes(b'new backend')
    (tmp_path / 'play-release-manifest.json').write_text(json.dumps({
        'revision': 'test', 'files': {'backend.py': hashlib.sha256(source.read_bytes()).hexdigest()},
    }))
    assert verify_files(tmp_path) == 'test'
    source.write_bytes(b'old backend')
    with pytest.raises(RuntimeError, match='Release file mismatch'):
        verify_files(tmp_path)


def test_release_verification_rejects_missing_delete_route():
    from urllib.error import HTTPError
    from io import BytesIO
    class Response(BytesIO):
        status = 200
    def request(req, **_):
        if req.method == 'DELETE':
            raise HTTPError(req.full_url, 405, 'missing route', {}, None)
        return Response(b'{"ok":true}')
    with patch('tools.verify_play_release.urlopen', request):
        with pytest.raises(RuntimeError, match='expected 401, received 405'):
            verify_routes('http://localhost')
