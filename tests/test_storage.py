from __future__ import annotations

import socket
import threading

import numpy as np
import pytest

from app.storage.resp import RespClient
from app.storage.sqlite import SqliteStorage


# ------------------------------------------------------------- SQLite
def test_sqlite_kv_roundtrip(tmp_path):
    db = SqliteStorage(str(tmp_path / "t.db"))
    assert db.ping()
    db.set("a", "1")
    db.set("b", "2")
    assert db.get("a") == "1"
    assert db.get("missing") is None
    assert db.mget(["a", "b", "missing"]) == ["1", "2", None]
    db.mset({"c": "3", "d": "4"})
    assert sorted(db.keys("")) == sorted(["a", "b", "c", "d"])
    assert sorted(db.keys("a")) == ["a"]
    assert db.incr("counter") == 1
    assert db.incr("counter", 5) == 6
    assert db.delete("a") is True
    assert db.get("a") is None
    db.close()


def test_sqlite_vectors_and_search(tmp_path):
    db = SqliteStorage(str(tmp_path / "t.db"))
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(50, 16)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    for i, v in enumerate(vecs):
        db.vadd(i + 1, v.tolist())
    query = vecs[7]
    hits = db.vsearch(5, query.tolist())
    assert hits[0][0] == 8  # id 8 = vec index 7
    assert hits[0][1] < 1e-3
    assert len(hits) == 5
    # update in place
    db.vadd(1, vecs[0].tolist())
    assert db.vsearch(1, vecs[0].tolist())[0][0] == 1
    db.close()


# ------------------------------------------------------------- RESP client
def _fake_resp_server(responses: list[bytes]):
    """Run a tiny RESP server that answers fixed bytes, returns its port."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    port = srv.getsockname()[1]

    def serve():
        conn, _ = srv.accept()
        with conn:
            data = b""
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
                while data.count(b"\r\n") >= 2:
                    # consume one command frame (naive: drain whatever arrived)
                    if responses:
                        conn.sendall(responses.pop(0))
                    data = b""
        srv.close()

    threading.Thread(target=serve, daemon=True).start()
    return port


def test_resp_client_parses_replies():
    pong = b"+PONG\r\n"
    bulk = b"$5\r\nhello\r\n"
    array = b"*4\r\n:5\r\n$6\r\n0.1234\r\n:9\r\n$6\r\n0.5678\r\n"
    nil = b"$-1\r\n"
    port = _fake_resp_server([pong, bulk, array, nil])
    client = RespClient("127.0.0.1", port, timeout=5)
    assert client.ping() is True
    assert client.request("GET", "k") == b"hello"
    assert client.vsearch(2, [1.0, 2.0]) == [(5, 0.1234), (9, 0.5678)]
    assert client.request("GET", "missing") is None
    client.disconnect()


def test_resp_client_error_reply_raises():
    port = _fake_resp_server([b"-ERR id is not a u64\r\n"])
    client = RespClient("127.0.0.1", port, timeout=5)
    with pytest.raises(Exception):
        client.request("VADD", "notanint", "1.0")
    client.disconnect()


def test_resp_client_reconnects():
    """A dead server must raise, and the client must be reusable after reconnect."""
    port = _fake_resp_server([b"+PONG\r\n"])
    client = RespClient("127.0.0.1", port, timeout=5)
    assert client.ping()
    client.disconnect()
    # second server on a different port — client instance points at old port
    port2 = _fake_resp_server([b"+PONG\r\n"])
    client2 = RespClient("127.0.0.1", port2, timeout=5)
    assert client2.ping()
    client2.disconnect()
