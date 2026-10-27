"""Cliente mínimo do Hugging Face (só biblioteca padrão).

Login por token de acesso: https://huggingface.co/settings/tokens
(um token do tipo "Read" basta).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from . import config

ENDPOINT = os.environ.get("FAROL_HF_ENDPOINT", "https://huggingface.co").rstrip("/")
UA = "Farol/0.1 (+local model manager)"
SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)


class HFError(Exception):
    def __init__(self, msg: str, status: int = 0):
        super().__init__(msg)
        self.status = status


def _req(path: str, token: str | None = None, timeout: float = 20) -> object:
    url = path if path.startswith("http") else f"{ENDPOINT}{path}"
    headers = {"User-Agent": UA, "Accept": "application/json"}
    tok = token if token is not None else config.get("hf_token")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise HFError("Token inválido ou expirado. Gere outro em huggingface.co/settings/tokens.", 401)
        if e.code == 403:
            raise HFError("Acesso negado. Este modelo pode exigir aceitar os termos na página dele.", 403)
        if e.code == 404:
            raise HFError("Modelo não encontrado no Hugging Face.", 404)
        raise HFError(f"Hugging Face respondeu {e.code}.", e.code)
    except urllib.error.URLError as e:
        raise HFError(f"Sem conexão com o Hugging Face ({e.reason}).")


# ------------------------------------------------------------------ conta

def login(token: str) -> dict:
    token = token.strip()
    if not token.startswith("hf_"):
        raise HFError("O token do Hugging Face começa com 'hf_'.")
    who = _req("/api/whoami-v2", token=token)
    config.set("hf_token", token)
    config.set("hf_user", {"name": who.get("name"), "fullname": who.get("fullname"),
                           "avatar": who.get("avatarUrl")})
    return config.get("hf_user")


def logout() -> None:
    config.delete("hf_token")
    config.delete("hf_user")


def account() -> dict | None:
    return config.get("hf_user")


# ----------------------------------------------------------------- busca

def search(query: str = "", limit: int = 24, sort: str = "downloads") -> list[dict]:
    params = {"filter": "gguf", "sort": sort, "direction": "-1", "limit": str(limit)}
    if query:
        params["search"] = query
    items = _req("/api/models?" + urllib.parse.urlencode(params))
    out = []
    for m in items:
        repo = m.get("id") or m.get("modelId")
        out.append({
            "repo": repo,
            "author": repo.split("/")[0] if repo else "",
            "name": repo.split("/")[-1] if repo else "",
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "updated": (m.get("lastModified") or "")[:10],
            "pipeline": m.get("pipeline_tag"),
            "gated": bool(m.get("gated")),
            "params_b": _params_from_name(repo or ""),
        })
    return out


def _params_from_name(name: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", name.replace("_", "-"))
    return float(m.group(1)) if m else None


def gguf_files(repo: str) -> list[dict]:
    """Lista os .gguf do repositório, juntando arquivos divididos em partes."""
    tree = _req(f"/api/models/{repo}/tree/main?recursive=1")
    groups: dict[str, dict] = {}
    for e in tree:
        path = e.get("path", "")
        if e.get("type") != "file" or not path.lower().endswith(".gguf"):
            continue
        low = path.lower()
        if "mmproj" in low or "imatrix" in low:
            continue  # arquivos auxiliares, não são o modelo
        size = (e.get("lfs") or {}).get("size") or e.get("size") or 0
        key = SPLIT_RE.sub(".gguf", path)
        g = groups.setdefault(key, {"name": key.split("/")[-1], "parts": [], "size": 0})
        g["parts"].append(path)
        g["size"] += size
    for g in groups.values():
        g["parts"].sort()
    return sorted(groups.values(), key=lambda g: g["size"])


def model_card(repo: str) -> dict:
    info = _req(f"/api/models/{repo}")
    card = info.get("cardData") or {}
    return {
        "repo": repo,
        "license": card.get("license"),
        "base_model": card.get("base_model"),
        "tags": [t for t in info.get("tags", []) if ":" not in t][:12],
        "gated": bool(info.get("gated")),
        "downloads": info.get("downloads", 0),
        "likes": info.get("likes", 0),
    }


# -------------------------------------------------------------- download

class Downloads:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def start(self, repo: str, parts: list[str], name: str) -> str:
        job_id = f"{repo}::{name}"
        with self.lock:
            if job_id in self.jobs and self.jobs[job_id]["status"] == "baixando":
                return job_id
            self.jobs[job_id] = {"id": job_id, "repo": repo, "name": name, "status": "baixando",
                                 "done": 0, "total": 0, "speed": 0.0, "error": None, "cancel": False}
        threading.Thread(target=self._run, args=(job_id, repo, parts), daemon=True).start()
        return job_id

    def cancel(self, job_id: str) -> None:
        if job_id in self.jobs:
            self.jobs[job_id]["cancel"] = True

    def list(self) -> list[dict]:
        return [{k: v for k, v in j.items() if k != "cancel"} for j in self.jobs.values()]

    def _run(self, job_id: str, repo: str, parts: list[str]) -> None:
        job = self.jobs[job_id]
        dest_dir = config.models_dir() / repo.replace("/", "__")
        dest_dir.mkdir(parents=True, exist_ok=True)
        token = config.get("hf_token")
        try:
            # tamanho total primeiro, para a barra de progresso fazer sentido
            sizes = []
            for p in parts:
                req = urllib.request.Request(self._url(repo, p), method="HEAD",
                                             headers=self._headers(token))
                with urllib.request.urlopen(req, timeout=30) as r:
                    sizes.append(int(r.headers.get("Content-Length") or 0))
            job["total"] = sum(sizes)
            base_done = 0
            for p, size in zip(parts, sizes):
                final = dest_dir / p.split("/")[-1]
                if final.exists() and final.stat().st_size == size:
                    base_done += size
                    job["done"] = base_done
                    continue
                self._fetch(job, repo, p, final, base_done, token)
                base_done += size
            job["status"] = "concluído"
            job["done"] = job["total"]
        except _Cancelled:
            job["status"] = "cancelado"
        except urllib.error.HTTPError as e:
            job["status"] = "erro"
            job["error"] = ("Este modelo exige login ou aceitar os termos no site do Hugging Face."
                            if e.code in (401, 403) else f"Erro HTTP {e.code}")
        except Exception as e:  # noqa: BLE001
            job["status"] = "erro"
            job["error"] = str(e)

    def _url(self, repo: str, path: str) -> str:
        return f"{ENDPOINT}/{repo}/resolve/main/{urllib.parse.quote(path)}"

    def _headers(self, token: str | None) -> dict:
        h = {"User-Agent": UA}
        if token:
            h["Authorization"] = f"Bearer {token}"
        return h

    def _fetch(self, job, repo, path, final, base_done, token) -> None:
        part = final.with_suffix(final.suffix + ".part")
        have = part.stat().st_size if part.exists() else 0
        headers = self._headers(token)
        if have:
            headers["Range"] = f"bytes={have}-"  # retoma de onde parou
        req = urllib.request.Request(self._url(repo, path), headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            if have and r.status != 206:
                have = 0
            mode = "ab" if have else "wb"
            t0, last_bytes = time.time(), have
            with open(part, mode) as f:
                while True:
                    if job["cancel"]:
                        raise _Cancelled()
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
                    job["done"] = base_done + have
                    dt = time.time() - t0
                    if dt > 1:
                        job["speed"] = (have - last_bytes) / dt
                        t0, last_bytes = time.time(), have
        part.replace(final)


class _Cancelled(Exception):
    pass


downloads = Downloads()
