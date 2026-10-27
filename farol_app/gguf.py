"""Leitor mínimo de metadados GGUF (camadas, contexto, cabeças de atenção).

Lê só o cabeçalho, sem carregar os pesos. Serve para o ajuste automático
saber quantas camadas o modelo tem e quanto ocupa o cache de contexto.
"""
from __future__ import annotations

import struct

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
            10: "<Q", 11: "<q", 12: "<d"}
WANTED = ("block_count", "context_length", "embedding_length",
          "attention.head_count", "attention.head_count_kv", "attention.key_length")


class _R:
    def __init__(self, f):
        self.f = f

    def take(self, fmt):
        n = struct.calcsize(fmt)
        return struct.unpack(fmt, self.f.read(n))[0]

    def string(self):
        n = self.take("<Q")
        return self.f.read(n).decode("utf-8", "replace")

    def value(self, t):
        if t in _SCALARS:
            return self.take(_SCALARS[t])
        if t == 8:
            return self.string()
        if t == 9:
            it, count = self.take("<I"), self.take("<Q")
            if it in _SCALARS:  # pula arrays numéricos sem ler item a item
                self.f.seek(struct.calcsize(_SCALARS[it]) * count, 1)
            else:
                for _ in range(count):
                    self.value(it)
            return None
        raise ValueError(f"tipo GGUF desconhecido: {t}")


def read_meta(path: str) -> dict:
    out: dict = {}
    with open(path, "rb") as f:
        r = _R(f)
        if f.read(4) != b"GGUF":
            raise ValueError("não é um arquivo GGUF")
        version = r.take("<I")
        if version < 2:
            raise ValueError("GGUF v1 não suportado")
        r.take("<Q")  # tensor_count
        kv_count = r.take("<Q")
        arch = None
        for _ in range(kv_count):
            key = r.string()
            val = r.value(r.take("<I"))
            if key == "general.architecture":
                arch = val
                out["arch"] = val
            elif key == "general.name":
                out["name"] = val
            elif arch and key.startswith(arch + "."):
                short = key[len(arch) + 1:]
                if short in WANTED:
                    out[short] = val
            if arch and all(k in out for k in ("block_count", "context_length", "embedding_length",
                                                 "attention.head_count", "attention.head_count_kv")):
                break
    return out


def kv_bytes_per_token(meta: dict) -> float | None:
    """Bytes do cache K+V por token de contexto (f16)."""
    try:
        layers = int(meta["block_count"])
        heads = int(meta["attention.head_count"])
        kv_heads = int(meta.get("attention.head_count_kv") or heads)
        head_dim = int(meta.get("attention.key_length") or meta["embedding_length"] // heads)
        return 2 * layers * kv_heads * head_dim * 2
    except (KeyError, TypeError, ZeroDivisionError, ValueError):
        return None
