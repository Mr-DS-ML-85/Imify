#!/usr/bin/env python3
"""Write a Netscape cookie file containing ONLY the domains the search
engines need (allowlist), so the bot browser never carries unrelated
sessions (password managers, mail, shops...).

usage: filter_cookies.py SRC DST
"""
import sys

ALLOW = (
    "google.com", "google.co", "gstatic.com", "youtube.com",  # Google consent/NID
    "bing.com", "microsoft.com", "live.com",
    "yandex.com", "yandex.ru", "ya.ru",
    "duckduckgo.com", "brave.com", "qwant.com", "startpage.com",
    "facebook.com", "fbcdn.net", "fbsbx.com", "instagram.com",
    "pinterest.com",
)


def allowed(domain: str) -> bool:
    d = domain.replace("#HttpOnly_", "").lstrip(".").lower()
    return any(d == a or d.endswith("." + a) or d.startswith(a + ".") for a in ALLOW)


def main(src: str, dst: str) -> None:
    kept = 0
    with open(src, errors="ignore") as fin, open(dst, "w") as fout:
        fout.write("# Netscape HTTP Cookie File (filtered allowlist for proxify)\n")
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 7 and allowed(parts[0]):
                fout.write(line if line.endswith("\n") else line + "\n")
                kept += 1
    print(f"kept {kept} cookies for {len(ALLOW)} allowed domains -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
