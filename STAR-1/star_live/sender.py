"""Blender addon'una UDP üzerinden iskelet verisi gönderir."""
from __future__ import annotations

import json
import socket
import struct
from typing import Any, Dict


class UDPSender:
    """
    65 KB UDP sınırını aşan JSON paketlerini chunk'lara bölerek gönderir.
    Blender addon'u aynı protokolü kullanarak paketleri yeniden birleştirir.

    Paket başlığı (6 byte, big-endian):
        [seq: 2B] [toplam_chunk: 2B] [chunk_indeksi: 2B]
    """

    MAX_CHUNK_SIZE = 60_000

    def __init__(self, host: str = "127.0.0.1", port: int = 7777) -> None:
        self.address = (host, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sequence = 0

    def send(self, data: Dict[str, Any]) -> None:
        payload = json.dumps(data, separators=(",", ":")).encode("utf-8")
        chunks  = [
            payload[i : i + self.MAX_CHUNK_SIZE]
            for i in range(0, len(payload), self.MAX_CHUNK_SIZE)
        ]
        seq   = self._sequence & 0xFFFF
        total = len(chunks)
        self._sequence += 1

        for idx, chunk in enumerate(chunks):
            header = struct.pack("!HHH", seq, total, idx)
            self._socket.sendto(header + chunk, self.address)

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> "UDPSender":
        return self

    def __exit__(self, *_) -> None:
        self.close()
