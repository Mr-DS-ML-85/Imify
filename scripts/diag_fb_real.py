#!/usr/bin/env python3
"""Query = the user's actual social-media profile pic; candidate = the new
photo/?fbid= page. Does the render path decode and match?"""
import os
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from app.config import Settings
from app.features import decode_image, extract_features
from app.matcher import Matcher
from app.models import ImageHit
from app.repo import Repository
from app.storage import SqliteStorage

settings = Settings(
    storage_backend="sqlite",
    sqlite_path="data/fb-real2.db",
    data_dir="data",
    cookie_file="gui-cookies.txt",
    proxify_fetch_url="http://127.0.0.1:8081",
    fetch_proxy_url="http://127.0.0.1:8888",
    web_verify_candidates=True,
    web_verify_concurrency=2,
)
settings.ensure_dirs()
repo = Repository(SqliteStorage(settings.sqlite_path))
matcher = Matcher(repo, settings)

qpath = "images/726002745_1391225456545660_7750183954756539772_n.jpg"
qraw = open(qpath, "rb").read()
qimg = decode_image(qraw)
print("query:", qpath, qimg.size if qimg else None)
if not qimg:
    sys.exit(1)
qf = extract_features(qimg, qraw)
print("query phash:", hex(qf["phash"]))

url = "https://www.facebook.com/photo/?fbid=1391220289879510&set=a.646142547720625"
hit = ImageHit(title="facebook", url=url, img_src=url, engine="bing")
results = matcher.verify_web_hits(qf, [hit], limit=2)
r0 = results[0] if results else None
if r0:
    print(f"verdict: {r0.match_type} score={r0.score} hamming={r0.hamming}")
    print(f"  url: {r0.url[:90]}")
    print(f"  page_url: {r0.page_url[:90]}")
else:
    print("no result")
