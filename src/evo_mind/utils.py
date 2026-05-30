"""Utility functions for evo-mind."""

from __future__ import annotations

import os
import time
import uuid


def uuid7() -> str:
    """Generate a time-sortable UUID7 string.

    UUID7: 48-bit Unix timestamp (ms) at front for sortability.
    Produces lexicographically sortable IDs ideal for B-tree primary keys in SQLite.
    """
    # Get timestamp in milliseconds (48 bits)
    timestamp_ms = int(time.time() * 1000)

    # Generate 10 random bytes
    rand = os.urandom(10)

    # Build 16-byte UUID:
    # bytes 0-5:  timestamp (48 bits, big-endian)
    # byte  6:    version 0x7 in high 4 bits + low 4 bits from rand[0]
    # byte  7:    rand[1]
    # byte  8:    variant 0b10 in high 2 bits + low 6 bits from rand[2]
    # bytes 9-15: rand[3:10] (7 bytes)

    ts_bytes = timestamp_ms.to_bytes(6, "big")

    return str(uuid.UUID(bytes=(
        ts_bytes[0:6]                                    # 6 bytes
        + bytes([0x70 | (rand[0] >> 4)])                 # byte 6: version
        + rand[1:2]                                      # byte 7: random
        + bytes([0x80 | (rand[2] & 0x3F)])               # byte 8: variant
        + rand[3:10]                                     # bytes 9-15: random (7 bytes)
    )))
