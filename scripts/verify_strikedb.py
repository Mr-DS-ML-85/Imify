"""Live check: VSEARCH over ±1-encoded hashes == Hamming-ordered ANN.

Run after starting StrikeDB on 127.0.0.1:6008:
    uv run python scripts/verify_strikedb.py
"""

import sys

from app.features import decode_hamming_from_distance, encode_hash_vector
from app.storage.resp import RespClient

A = 0xFFFFFFFFFFFFFFFF  # all 1s
B = 0x0000000000000000  # all 0s
C = 0xAAAAAAAAAAAAAAAA  # alternating

# Expected hamming distances from A:
#   A vs A -> 0
#   A vs C -> 32 (half the bits differ)
#   A vs B -> 64 (all 64 bits differ)
EXPECTED = {2: 64, 3: 32}


def main() -> int:
    c = RespClient("127.0.0.1", 6008)
    if not c.ping():
        print("FAIL: StrikeDB not reachable on 127.0.0.1:6008")
        return 1

    c.vadd(1, encode_hash_vector(A))
    c.vadd(2, encode_hash_vector(B))
    c.vadd(3, encode_hash_vector(C))

    hits = c.vsearch(3, encode_hash_vector(A))
    print("VSEARCH result (id, dist):", hits)
    assert hits[0][0] == 1, "self-match should rank first"
    assert abs(hits[0][1]) < 0.01, "self-match distance should be ~0"
    for vid, dist in hits[1:]:
        reported = decode_hamming_from_distance(dist)
        print(f"  id {vid}: dist={dist:.4f} -> reported hamming={reported}")
        assert abs(reported - EXPECTED[vid]) <= 1, f"expected hamming {EXPECTED[vid]}, got {reported}"
    print("OK: Hamming-ordered ANN works on the real StrikeDB binary (dist == hamming/32)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
