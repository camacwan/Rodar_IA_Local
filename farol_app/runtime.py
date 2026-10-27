"""Motor que roda os modelos (llama.cpp): instalação, ajuste automático e execução.

Motores aceitos, nesta ordem:
  1. binário `llama-server` do llama.cpp (PATH ou FAROL_LLAMA_SERVER)
  2. llama-cpp-python no ambiente próprio do Farol (~/.farol/venv), instalado pelo app
  3. llama-cpp-python no Python que está rodando o Farol
Todos expõem uma API compatível com a da OpenAI em /v1.
"""
from __future__ import annotations

import collections
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config, gguf

ENGINE_PORT = int(os.environ.get("FAROL_ENGINE_PORT", "8081"))
GB = 1024 ** 3
WHEELS = "https://abetlen.github.io/llama-cpp-python/whl"


# ------------------------------------------------------------ onde está o motor

def venv_dir() -> Path:
    return config.HOME / "venv"


def venv_python() -> Path:
    v = venv_dir()
    return v / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _has_llama_cpp(python: Path) -> bool:
    if not python.exists():
        return False
    r = subprocess.run([str(python), "-c", "import llama_cpp, sys; print(llama_cpp.__version__)"],
                       capture_output=True, text=True, timeout=30)
    return r.returncode == 0


_engine_cache: dict = {}


def find_engine(refresh: bool = False) -> dict | None:
    if _engine_cache and not refresh:
        return _engine_cache.get("v")
    found = None
    custom = os.environ.get("FAROL_LLAMA_SERVER") or config.get("llama_server_path")
    for cand in [custom, shutil.which("llama-server"), shutil.which("llama-server.exe")]:
        if cand and Path(cand).exists():
            found = {"kind": "llama-server", "path": cand}
            break
    if not found and _has_llama_cpp(venv_python()):
        found = {"kind": "llama-cpp-python", "path": str(venv_python())}
    if not found and importlib.util.find_spec("llama_cpp"):
        found = {"kind": "llama-cpp-python", "path": sys.executable}
    _engine_cache["v"] = found
    return found


# ------------------------------------------------------------------ instalação

def _cuda_version() -> tuple[int, int] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def install_plan() -> dict:
    """Escolhe o pacote pré-compilado certo para esta máquina (evita compilar)."""
    cuda = _cuda_version()
    mac_arm = platform.system() == "Darwin" and platform.machine() == "arm64"
    if cuda and cuda[0] >= 12:
        minor = min(cuda[1], 5)  # há pacotes prontos de cu121 a cu125
        minor = max(minor, 1)
        return {"variant": f"NVIDIA CUDA 12.{minor}", "index": f"{WHEELS}/cu12{minor}", "accel": "gpu"}
    if mac_arm:
        return {"variant": "Apple Metal", "index": f"{WHEELS}/metal", "accel": "gpu"}
    return {"variant": "CPU", "index": f"{WHEELS}/cpu", "accel": "cpu"}


GH_RELEASES = os.environ.get("FAROL_GH_API", "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")


def _has_vulkan() -> bool:
    if os.name == "nt":
        return Path(os.environ.get("WINDIR", r"C:\Windows"), "System32", "vulkan-1.dll").exists()
    return bool(shutil.which("vulkaninfo")) or any(Path(p).exists() for p in (
        "/usr/lib/x86_64-linux-gnu/libvulkan.so.1", "/usr/lib/libvulkan.so.1"))


def pick_assets(assets: list[dict], system: str, machine: str,
                cuda: tuple[int, int] | None, vulkan: bool) -> tuple[str, list[dict]]:
    """Escolhe o(s) arquivo(s) do release oficial do llama.cpp para esta máquina.

    Os nomes seguem o padrão llama-<tag>-bin-<plataforma>.zip (ou .tar.gz).
    Devolve (descrição, [assets]). Lista vazia = nada compatível.
    """
    names = {a["name"].lower(): a for a in assets if a["name"].lower().endswith((".zip", ".tar.gz"))}
    arch = "arm64" if machine.lower() in ("arm64", "aarch64") else "x64"

    def find(*must: str, exclude: tuple[str, ...] = ()) -> dict | None:
        for n, a in sorted(names.items()):
            if n.startswith("llama-") and all(m in n for m in must) and not any(x in n for x in exclude):
                return a
        return None

    if system == "Darwin":
        a = find("bin-macos-" + arch)
        return ("macOS " + ("Apple Silicon (Metal)" if arch == "arm64" else "Intel"), [a] if a else [])

    if system == "Windows":
        if cuda:
            # o maior CUDA que o driver suporta
            best = None
            for n, a in names.items():
                m = re.search(r"bin-win-cuda-?(\d+)\.(\d+)-" + arch, n)
                if n.startswith("llama-") and m:
                    v = (int(m.group(1)), int(m.group(2)))
                    if v <= cuda and (best is None or v > best[0]):
                        best = (v, a)
            if best:
                v = f"{best[0][0]}.{best[0][1]}"
                runtime = next((a for n, a in names.items()
                                if n.startswith("cudart-") and f"cuda-{v}" in n and arch in n), None)
                return (f"Windows NVIDIA CUDA {v}", [best[1]] + ([runtime] if runtime else []))
        if vulkan:
            a = find("bin-win-vulkan-" + arch)
            if a:
                return ("Windows Vulkan (GPU AMD, Intel ou NVIDIA)", [a])
        a = find("bin-win-cpu-" + arch) or find("bin-win-" + arch, exclude=("cuda", "vulkan", "hip", "sycl", "opencl"))
        return ("Windows CPU", [a] if a else [])

    # Linux
    if vulkan:
        a = find("bin-ubuntu-vulkan-" + arch)
        if a:
            return ("Linux Vulkan", [a])
    a = find("bin-ubuntu-" + arch, exclude=("vulkan", "cuda", "rocm", "hip", "sycl", "openvino", "s390x"))
    return ("Linux CPU", [a] if a else [])


def install_options() -> list[dict]:
    plan = install_plan()
    return [
        {"kind": "binary", "title": "Motor oficial do llama.cpp", "recommended": True,
         "desc": "Programa pronto do projeto llama.cpp, sempre atualizado. Suporta os modelos mais novos "
                 "e GPUs NVIDIA, AMD, Intel e Apple. Não usa Python."},
        {"kind": "pip", "title": "llama-cpp-python", "recommended": False,
         "desc": f"Pacote Python ({plan['variant']}). Pode ficar atrás do llama.cpp e não abrir modelos recentes."},
    ]


class Installer:
    def __init__(self) -> None:
        self.state = {"status": "parado", "log": [], "plan": None, "kind": None}

    def status(self) -> dict:
        return {**self.state, "log": self.state["log"][-40:], "options": install_options()}

    def start(self, kind: str = "binary") -> dict:
        if self.state["status"] == "instalando":
            return self.status()
        plan = install_plan() if kind == "pip" else {"variant": "procurando a versão certa…"}
        self.state = {"status": "instalando", "log": [], "plan": plan, "kind": kind}
        target = self._run_pip if kind == "pip" else self._run_binary
        threading.Thread(target=self._guard, args=(target,), daemon=True).start()
        return self.status()

    def _say(self, line: str) -> None:
        self.state["log"].append(line.rstrip())

    def _guard(self, fn) -> None:
        try:
            ok = fn()
            self.state["status"] = "concluído" if ok else "erro"
            if ok:
                find_engine(refresh=True)
                self._say("Motor instalado. Já dá para iniciar modelos.")
        except Exception as e:  # noqa: BLE001
            self.state["status"] = "erro"
            self._say(f"Erro: {e}")

    # -------- motor oficial (binário)
    def _run_binary(self) -> bool:
        self._say("Consultando a versão mais recente do llama.cpp no GitHub…")
        req = urllib.request.Request(GH_RELEASES, headers={"User-Agent": "Farol", "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            rel = json.loads(r.read())
        tag = rel.get("tag_name", "latest")
        system, machine = platform.system(), platform.machine()
        choices = [pick_assets(rel.get("assets", []), system, machine, _cuda_version(), _has_vulkan())]
        cpu_choice = pick_assets(rel.get("assets", []), system, machine, None, False)
        if cpu_choice[1] and cpu_choice[1] != choices[0][1]:
            choices.append(cpu_choice)  # plano B: versão só CPU, que abre em qualquer máquina
        for n, (desc, assets) in enumerate(choices):
            self.state["plan"] = {"variant": f"{desc} · versão {tag}"}
            if not assets:
                continue
            if n:
                self._say(f"Tentando a versão {desc}…")
            try:
                if self._install_assets(tag, desc, assets):
                    return True
            except Exception as e:  # noqa: BLE001
                self._say(f"Falhou: {e}")
        self._say(f"Não consegui instalar um motor pronto para {system} {machine} (versão {tag}).")
        self._say("Baixe manualmente em github.com/ggml-org/llama.cpp/releases ou use a opção llama-cpp-python.")
        return False

    def _install_assets(self, tag: str, desc: str, assets: list[dict]) -> bool:
        import tarfile
        import zipfile
        dest = config.HOME / "llama.cpp" / f"{tag}-{re.sub(r'[^a-z0-9]+', '-', desc.lower())[:30]}"
        dest.mkdir(parents=True, exist_ok=True)
        for a in assets:
            self._say(f"Baixando {a['name']} ({a.get('size', 0) / 1024**2:.0f} MB)…")
            archive = dest / a["name"]
            self._download(a["browser_download_url"], archive, a.get("size") or 0)
            self._say("Extraindo…")
            if a["name"].lower().endswith(".zip"):
                with zipfile.ZipFile(archive) as z:
                    z.extractall(dest)
            else:
                with tarfile.open(archive) as t:
                    t.extractall(dest, filter="data") if sys.version_info >= (3, 12) else t.extractall(dest)
            archive.unlink(missing_ok=True)
        exe = "llama-server.exe" if os.name == "nt" else "llama-server"
        found = next(iter(sorted(dest.rglob(exe))), None)
        if not found:
            self._say("O pacote não trouxe o llama-server.")
            return False
        if os.name == "nt":
            # as DLLs do CUDA precisam ficar ao lado do executável
            for dll in dest.rglob("*.dll"):
                if dll.parent != found.parent and not (found.parent / dll.name).exists():
                    shutil.copy2(dll, found.parent / dll.name)
        else:
            for f in found.parent.iterdir():
                if f.is_file() and "." not in f.name:
                    f.chmod(0o755)
        r = subprocess.run([str(found), "--version"], capture_output=True, text=True, timeout=60)
        for line in (r.stdout + r.stderr).strip().splitlines()[-4:]:
            self._say(line)
        if r.returncode != 0:
            self._say("Esta versão não abriu nesta máquina" +
                      (" (talvez o driver da NVIDIA esteja desatualizado)." if "cuda" in desc.lower() else "."))
            return False
        config.set("llama_server_path", str(found))
        self._say(f"Instalado em {found.parent}")
        return True

    def _download(self, url: str, path: Path, size: int) -> None:
        req = urllib.request.Request(url, headers={"User-Agent": "Farol"})
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
            done, last = 0, 0
            while chunk := r.read(1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if size and done - last > size / 5:
                    last = done
                    self._say(f"  {done / size * 100:.0f}%")

    # -------- llama-cpp-python (pip)
    def _pip(self, python: Path, extra: list[str]) -> int:
        cmd = [str(python), "-m", "pip", "install", "--upgrade", "--prefer-binary",
               "llama-cpp-python[server]", *extra]
        self._say("$ " + " ".join(cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:  # type: ignore[union-attr]
            if line.strip():
                self._say(line)
        return p.wait()

    def _run_pip(self) -> bool:
        plan = self.state["plan"]
        py = venv_python()
        if not py.exists():
            self._say("Criando ambiente isolado em " + str(venv_dir()))
            r = subprocess.run([sys.executable, "-m", "venv", str(venv_dir())], capture_output=True, text=True)
            if r.returncode != 0:
                self._say(r.stderr.strip())
                self._say("Não deu para criar o ambiente isolado; instalando no Python atual.")
                py = Path(sys.executable)
        self._say(f"Pacote escolhido para esta máquina: {plan['variant']}")
        code = self._pip(py, ["--extra-index-url", plan["index"]])
        if code != 0 and plan["accel"] == "gpu":
            self._say("Pacote acelerado indisponível para esta versão do Python. Tentando a versão CPU.")
            code = self._pip(py, ["--extra-index-url", f"{WHEELS}/cpu"])
        if code != 0:
            self._say("A instalação falhou. Use a opção Motor oficial do llama.cpp.")
        return code == 0


installer = Installer()


# ---------------------------------------------------------- ajuste automático

def tune(model_path: str, hw: dict, bench: dict | None, max_ctx: int | None = None) -> dict:
    """Escolhe camadas na GPU, contexto, threads e flash attention para esta máquina."""
    files = _parts(Path(model_path))
    size_gb = sum(f.stat().st_size for f in files) / GB
    try:
        meta = gguf.read_meta(model_path)
    except Exception:
        meta = {}
    layers = int(meta.get("block_count") or 32)
    train_ctx = int(meta.get("context_length") or 8192)
    kv_tok = gguf.kv_bytes_per_token(meta)
    kv_gb = (lambda c: c * kv_tok / GB) if kv_tok else (lambda c: size_gb * 0.04 * c / 4096)
    overhead = 0.6

    ram_usable = max(1.0, hw["ram_total_gb"] - 2.5)
    gpu = max(hw.get("gpus") or [], key=lambda g: g.get("vram_gb") or 0, default=None)
    unified = bool(gpu and gpu.get("unified"))
    vram = (gpu["vram_gb"] if unified else (gpu["vram_gb"] or 0) - 0.5) if gpu and gpu.get("vram_gb") else 0

    cands = [c for c in (32768, 16384, 8192, 4096, 2048) if c <= min(train_ctx, max_ctx or 32768)] or [2048]
    # contexto padrão: até 8k (mais que isso só se pedido), o resto vira velocidade
    default_cap = 8192 if not max_ctx else max_ctx
    cands = [c for c in cands if c <= default_cap] or [cands[-1]]
    notes: list[str] = []

    def need(c: int) -> float:
        return size_gb + kv_gb(c) + overhead

    if vram and need(cands[-1]) <= vram:
        ctx = next(c for c in cands if need(c) <= vram)
        ngl = layers + 1
        notes.append(f"O modelo inteiro cabe na {'memória unificada' if unified else 'GPU'}: todas as {layers} camadas vão para lá.")
    elif vram > 1.5 and not unified:
        ctx = next((c for c in cands if need(c) <= vram + ram_usable), cands[-1])
        frac = max(0.0, (vram - kv_gb(ctx) - overhead) / size_gb)
        ngl = max(0, min(layers, int(layers * frac)))
        notes.append(f"Não cabe todo na GPU: {ngl} de {layers} camadas na GPU, o resto na RAM.")
    else:
        ctx = next((c for c in cands if need(c) <= ram_usable), cands[-1])
        ngl = 0
        notes.append("Sem GPU utilizável: roda na CPU usando a RAM.")

    cores = hw.get("cores") or os.cpu_count() or 4
    threads = cores if ngl <= layers // 2 else max(2, min(cores, 4))
    notes.append(f"Contexto de {ctx:,} tokens".replace(",", ".") +
                 (" (limite do próprio modelo)." if ctx == train_ctx else " — memória que sobra vai para velocidade."))
    notes.append(f"{threads} threads para gerar texto (núcleos físicos) e {hw.get('threads', cores)} para ler o prompt.")
    flash = bool(gpu) and ngl > 0
    if flash:
        notes.append("Flash attention ligado: menos memória no contexto e mais velocidade na GPU.")
    return {
        "n_gpu_layers": ngl if ngl <= layers else 999,
        "layers": layers,
        "n_ctx": ctx,
        "n_threads": threads,
        "n_threads_batch": hw.get("threads", cores),
        "n_batch": 512 if (ngl or ram_usable > 6) else 256,
        "flash_attn": flash,
        "model_ctx_max": train_ctx,
        "arch": meta.get("arch"),
        "notes": notes,
    }


def _parts(p: Path) -> list[Path]:
    if "-00001-of-" in p.name:
        return sorted(p.parent.glob(p.name.split("-00001-of-")[0] + "*.gguf"))
    return [p]


def local_models() -> list[dict]:
    out = []
    for f in sorted(config.models_dir().glob("*/*.gguf")):
        if "-of-" in f.name and "-00001-of-" not in f.name:
            continue  # só a primeira parte de modelos divididos
        out.append({"path": str(f), "repo": f.parent.name.replace("__", "/"), "name": f.name,
                    "size": sum(s.stat().st_size for s in _parts(f))})
    return out


# ------------------------------------------------------------------- execução

class Engine:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.model: str | None = None
        self.settings: dict | None = None
        self.log: collections.deque[str] = collections.deque(maxlen=300)
        self.lock = threading.RLock()
        self.gen = 0
        self.failed = False
        self.attempts: list[dict] = []

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{ENGINE_PORT}"

    def status(self) -> dict:
        running = self.proc is not None and self.proc.poll() is None
        return {
            "engine": find_engine(),
            "running": running,
            "ready": running and self._healthy(),
            "crashed": self.failed,
            "model": self.model if running else None,
            "settings": self.settings if running else None,
            "attempts": self.attempts,
            "port": ENGINE_PORT,
            "log_tail": list(self.log)[-(40 if self.failed else 15):],
        }

    def _healthy(self) -> bool:
        # llama-server: /health dá 503 enquanto carrega; llama-cpp-python não tem /health
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=1.5) as r:
                return r.status == 200
        except urllib.error.HTTPError as e:
            if e.code != 404:
                return False
        except Exception:
            return False
        try:
            with urllib.request.urlopen(self.base_url + "/v1/models", timeout=1.5) as r:
                return r.status == 200
        except Exception:
            return False

    def _cmd(self, eng: dict, path: str, s: dict) -> list[str]:
        if eng["kind"] == "llama-server":
            return [eng["path"], "-m", path, "--host", "127.0.0.1", "--port", str(ENGINE_PORT),
                    "-c", str(s["n_ctx"]), "-ngl", str(s["n_gpu_layers"]), "-t", str(s["n_threads"]),
                    "-tb", str(s["n_threads_batch"]), "-b", str(s["n_batch"]), "--jinja"]
        cmd = [eng["path"], "-m", "llama_cpp.server", "--model", path, "--host", "127.0.0.1",
               "--port", str(ENGINE_PORT), "--n_ctx", str(s["n_ctx"]),
               "--n_gpu_layers", str(-1 if s["n_gpu_layers"] >= 999 else s["n_gpu_layers"]),
               "--n_threads", str(s["n_threads"]), "--n_threads_batch", str(s["n_threads_batch"]),
               "--n_batch", str(s["n_batch"])]
        if s.get("flash_attn"):
            cmd += ["--flash_attn", "true"]
        return cmd

    @staticmethod
    def _fallbacks(s: dict) -> list[tuple[str, dict]]:
        """Configurações cada vez mais conservadoras, para quando o motor cai ao carregar."""
        out = [("Configuração automática", dict(s))]
        if s.get("flash_attn"):
            out.append(("Sem flash attention", {**s, "flash_attn": False}))
        if s.get("n_gpu_layers"):
            out.append(("Só CPU, sem GPU", {**s, "flash_attn": False, "n_gpu_layers": 0,
                                            "notes": s["notes"] + ["A GPU travou o motor; rodando só na CPU."]}))
        if s.get("n_ctx", 0) > 2048:
            out.append(("Contexto mínimo (2.048)", {**out[-1][1], "n_ctx": 2048, "n_batch": 256,
                                                    "notes": out[-1][1]["notes"] + ["Contexto reduzido para 2.048 tokens."]}))
        return out

    def start(self, model_path: str, hw: dict, bench: dict | None, overrides: dict | None = None) -> dict:
        eng = find_engine(refresh=True)
        if not eng:
            raise RuntimeError("Nenhum motor instalado. Clique em \"Instalar motor\" em Meus modelos.")
        if not Path(model_path).exists():
            raise RuntimeError("Arquivo do modelo não encontrado.")
        s = tune(model_path, hw, bench, (overrides or {}).get("n_ctx"))
        for k, v in (overrides or {}).items():
            if v not in (None, "") and k in s:
                s[k] = type(s[k])(v) if not isinstance(s[k], bool) else bool(v)
        # ajuste manual = só a configuração pedida; automático = tenta alternativas se cair
        profiles = [("Ajuste manual", s)] if overrides else self._fallbacks(s)
        if eng["kind"] == "llama-server":  # o binário decide o flash attention sozinho
            profiles = [p for p in profiles if p[0] != "Sem flash attention"]
        with self.lock:
            self.stop()
            self.log.clear()
            self.gen += 1
            self.failed = False
            self.attempts = []
            self._launch(eng, model_path, profiles[0])
            threading.Thread(target=self._watch, args=(self.gen, eng, model_path, profiles), daemon=True).start()
        return self.status()

    def _launch(self, eng: dict, path: str, profile: tuple[str, dict]) -> None:
        label, s = profile
        self.attempts.append({"label": label, "result": "tentando"})
        self.proc = subprocess.Popen(self._cmd(eng, path, s), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1,
                                     encoding="utf-8", errors="replace")
        self.model, self.settings = path, s
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()

    def _watch(self, gen: int, eng: dict, path: str, profiles: list) -> None:
        i = 0
        while gen == self.gen:
            proc = self.proc
            if proc is None:
                return
            if proc.poll() is not None:
                with self.lock:
                    if gen != self.gen:
                        return
                    self.attempts[i]["result"] = "falhou"
                    i += 1
                    if i >= len(profiles):
                        self.failed = True
                        return
                    time.sleep(0.5)  # deixa o log da falha terminar de chegar
                    self.log.append(f"--- Tentativa {i + 1}: {profiles[i][0]} ---")
                    self._launch(eng, path, profiles[i])
                continue
            if self._healthy():
                self.attempts[i]["result"] = "ok"
                return
            time.sleep(1)

    def _pump(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            self.log.append(line.rstrip())

    def stop(self) -> None:
        self.gen += 1
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        self.model = None
        self.failed = False

    def chat(self, messages: list[dict], max_tokens: int = 2048, temperature: float = 0.2) -> dict:
        body = json.dumps({"messages": messages, "max_tokens": max_tokens,
                           "temperature": temperature, "stream": False}).encode()
        req = urllib.request.Request(self.base_url + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())

    def speed_test(self, max_tokens: int = 128) -> dict:
        if not self.status()["ready"]:
            raise RuntimeError("Inicie um modelo e espere ele ficar pronto antes do teste.")
        t0 = time.perf_counter()
        data = self.chat([{"role": "user", "content":
                           "Explique em um parágrafo o que é inflação e por que ela importa."}],
                         max_tokens=max_tokens, temperature=0.7)
        dt = time.perf_counter() - t0
        usage, timings = data.get("usage") or {}, data.get("timings") or {}
        gen = usage.get("completion_tokens") or max_tokens
        result = {
            "tokens_per_s": round(timings.get("predicted_per_second") or gen / dt, 1),
            "tokens": gen, "seconds": round(dt, 1), "model": Path(self.model or "").name,
        }
        hist = config.get("speed_tests", {})
        hist[result["model"]] = {"tokens_per_s": result["tokens_per_s"], "at": time.strftime("%Y-%m-%d %H:%M")}
        config.set("speed_tests", hist)
        return result


engine = Engine()
