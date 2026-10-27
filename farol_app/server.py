"""Servidor HTTP local: interface web + API do app + proxy /v1 para o motor."""
from __future__ import annotations

import json
import threading
import traceback
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import agent as agent_mod
from . import config, estimator, hardware, hf
from .agent import agent
from .runtime import engine, find_engine, installer, local_models, tune

STATIC = Path(__file__).parent / "static"
GB = 1024 ** 3
_hw_cache: dict = {}
_bench_lock = threading.Lock()


def _hw(refresh: bool = False) -> dict:
    if refresh or not _hw_cache:
        _hw_cache.clear()
        _hw_cache.update(hardware.detect())
    return _hw_cache


def _with_estimate(size_bytes: int, name: str) -> dict:
    q = estimator.parse_quant(name)
    label, desc = estimator.quant_info(q)
    est = estimator.estimate(size_bytes / GB, _hw(), config.get("benchmark"))
    return {"quant": q, "quant_label": label, "quant_desc": desc,
            "size_gb": round(size_bytes / GB, 2), **est}


class Handler(BaseHTTPRequestHandler):
    server_version = "Farol/0.1"

    def log_message(self, fmt, *args):  # silencioso no terminal
        pass

    # ------------------------------------------------------------ helpers
    def _json(self, data, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n) or b"{}")

    def _query(self) -> dict:
        return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).items()}

    def _route(self) -> str:
        return urllib.parse.urlparse(self.path).path

    def _safe(self, fn) -> None:
        try:
            fn()
        except hf.HFError as e:
            self._json({"error": str(e)}, e.status or 502)
        except (RuntimeError, ValueError) as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._json({"error": f"Erro interno: {e}"}, 500)

    # ------------------------------------------------------------- verbos
    def do_GET(self):
        r = self._route()
        if r.startswith("/v1/"):
            return self._proxy("GET")
        if r in ("/", "/index.html"):
            body = (STATIC / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._safe(lambda: self._get(r))

    def do_POST(self):
        r = self._route()
        if r.startswith("/v1/"):
            return self._proxy("POST")
        self._safe(lambda: self._post(r))

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    # ---------------------------------------------------------------- GET
    def _get(self, r: str) -> None:
        q = self._query()
        if r == "/api/hardware":
            hw, bench = _hw(q.get("refresh") == "1"), config.get("benchmark")
            tiers = []
            for b in (1, 3, 4, 8, 14, 24, 32, 70):  # Q4_K_M ≈ 0,6 GB por bilhão de parâmetros
                est = estimator.estimate(b * 0.6, hw, bench)
                tiers.append({"params_b": b, "size_gb": round(b * 0.6, 1), **est})
            self._json({"hardware": hw, "benchmark": bench, "engine": find_engine(), "tiers": tiers})
        elif r == "/api/account":
            self._json({"account": hf.account()})
        elif r == "/api/search":
            results = hf.search(q.get("q", ""), limit=int(q.get("limit", 24)))
            # estimativa rápida pelo nº de parâmetros (Q4_K_M ≈ 0,6 GB por bilhão)
            for m in results:
                if m["params_b"]:
                    est = estimator.estimate(m["params_b"] * 0.6, _hw(), config.get("benchmark"))
                    m["light"] = est["light"]
                    m["tokens_per_s"] = est["tokens_per_s"]
            self._json({"results": results})
        elif r == "/api/model":
            repo = q["repo"]
            files = hf.gguf_files(repo)
            card = hf.model_card(repo)
            for f in files:
                f.update(_with_estimate(f["size"], f["name"]))
            best = next((f for f in reversed(files) if f["light"] == "verde"), None) \
                or next((f for f in reversed(files) if f["light"] == "amarelo"), None)
            if best:
                best["recommended"] = True
            self._json({"card": card, "files": files})
        elif r == "/api/downloads":
            self._json({"jobs": hf.downloads.list()})
        elif r == "/api/local":
            tests = config.get("speed_tests", {})
            items = []
            for m in local_models():
                m.update(_with_estimate(m["size"], m["name"]))
                m["measured"] = tests.get(m["name"])
                items.append(m)
            self._json({"models": items, "dir": str(config.models_dir())})
        elif r == "/api/engine":
            self._json(engine.status())
        elif r == "/api/engine/install":
            self._json(installer.status())
        elif r == "/api/engine/tune":
            p = Path(q["path"]).resolve()
            if config.models_dir().resolve() not in p.parents:
                raise ValueError("Modelo fora da pasta do Farol.")
            self._json(tune(str(p), _hw(), config.get("benchmark")))
        elif r == "/api/agent":
            self._json(agent.state(int(q.get("since", 0))))
        elif r == "/api/agent/models":
            self._json(agent_mod.public_role_models())
        elif r == "/api/agent/file":
            self._json({"content": agent.read_file(q["path"])})
        else:
            self._json({"error": "rota não encontrada"}, 404)

    # --------------------------------------------------------------- POST
    def _post(self, r: str) -> None:
        b = self._body()
        if r == "/api/benchmark":
            if not _bench_lock.acquire(blocking=False):
                raise RuntimeError("Já existe um teste rodando.")
            try:
                res = hardware.benchmark(float(b.get("seconds", 2)))
            finally:
                _bench_lock.release()
            config.set("benchmark", res)
            _hw(refresh=True)
            self._json({"benchmark": res, "hardware": _hw()})
        elif r == "/api/login":
            self._json({"account": hf.login(b.get("token", ""))})
        elif r == "/api/logout":
            hf.logout()
            self._json({"ok": True})
        elif r == "/api/download":
            parts = b.get("parts") or [b["name"]]
            job = hf.downloads.start(b["repo"], parts, b["name"])
            self._json({"job": job})
        elif r == "/api/download/cancel":
            hf.downloads.cancel(b["id"])
            self._json({"ok": True})
        elif r == "/api/local/delete":
            p = Path(b["path"]).resolve()
            root = config.models_dir().resolve()
            if root not in p.parents:
                raise ValueError("Só é possível apagar modelos da pasta do Farol.")
            if engine.model and Path(engine.model).resolve() == p:
                engine.stop()
            stem = p.name.split("-00001-of-")[0] if "-00001-of-" in p.name else None
            for f in (p.parent.glob(stem + "*.gguf") if stem else [p]):
                f.unlink(missing_ok=True)
            self._json({"ok": True})
        elif r == "/api/engine/start":
            self._json(engine.start(b["path"], _hw(), config.get("benchmark"), b.get("overrides")))
        elif r == "/api/engine/install":
            self._json(installer.start(b.get("kind", "binary")))
        elif r == "/api/agent/start":
            self._json(agent.start(b.get("workspace", ""), b.get("task", ""), int(b.get("max_rounds", 4))))
        elif r == "/api/agent/stop":
            agent.stop()
            self._json({"ok": True})
        elif r == "/api/agent/models":
            current = agent_mod.role_models()
            incoming = b.get("models", {})
            for role, m in incoming.items():  # chave mascarada na tela = manter a salva
                if m.get("api_key", "").startswith("…") and role in current:
                    m["api_key"] = current[role].get("api_key", "")
            self._json(agent_mod.save_role_models(incoming))
        elif r == "/api/engine/stop":
            engine.stop()
            self._json(engine.status())
        elif r == "/api/engine/test":
            self._json(engine.speed_test(int(b.get("max_tokens", 128))))
        else:
            self._json({"error": "rota não encontrada"}, 404)

    # --------------------------------------------------------------- proxy
    def _proxy(self, method: str) -> None:
        """Repassa /v1/* para o motor. Seus projetos usam http://localhost:PORTA/v1."""
        n = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(n) if n else None
        req = urllib.request.Request(engine.base_url + self.path, data=data, method=method,
                                     headers={"Content-Type": self.headers.get("Content-Type", "application/json")})
        try:
            up = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            up = e
        except Exception:
            self._json({"error": {"message": "Nenhum modelo carregado. Inicie um na aba Meus modelos."}}, 503)
            return
        with up:
            self.send_response(up.status if hasattr(up, "status") else up.code)
            ctype = up.headers.get("Content-Type", "application/json")
            self.send_header("Content-Type", ctype)
            self._cors()
            streaming = "event-stream" in ctype
            if streaming:
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                while chunk := up.read1(4096) if hasattr(up, "read1") else up.read(4096):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            else:
                body = up.read()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)


def serve(host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd
