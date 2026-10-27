"""Agente de programação em equipe (estilo Codex), pensado para máquinas modestas.

Fluxo:
  1. Arquiteto  — transforma o pedido numa especificação curta e lista os arquivos.
  2. Testador   — escreve os testes a partir da especificação, ANTES do código.
  3. Desenvolvedor — escreve o código para passar nos testes.
  4. Executor   — roda os testes de verdade (não é um modelo; é o Python).
  5. Se falhar, o desenvolvedor recebe o erro real e corrige. Se ele achar que o
     teste está errado, pode contestar e o testador revisa.

Por que funciona com modelos pequenos: cada papel recebe uma tarefa curta e
o "juiz" é a execução real dos testes, não a opinião de outro modelo.

Segurança: o código gerado roda no seu computador. O agente só escreve
dentro da pasta de trabalho escolhida e só executa os testes (sem comandos
livres), com tempo limite.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from . import config
from .runtime import engine

ROLES = {
    "arquiteto": "Arquiteto",
    "testador": "Testador",
    "desenvolvedor": "Desenvolvedor",
}
DEFAULT_MODELS = {r: {"type": "local"} for r in ROLES}

FILE_RE = re.compile(
    r"(?:\*\*|#+\s*)?(?:ARQUIVO|FILE)\s*:\s*`?([^\n`*]+?)`?(?:\*\*)?\s*\n+```[\w+.-]*\n(.*?)\n```",
    re.S | re.I,
)
FORMAT = (
    "Para CADA arquivo, use exatamente este formato, com o conteúdo COMPLETO do arquivo:\n"
    "ARQUIVO: caminho/relativo.py\n```python\n# código\n```\n"
    "Não escreva explicações longas fora dos blocos."
)


def role_models() -> dict:
    saved = config.get("agent_models") or {}
    return {r: saved.get(r) or DEFAULT_MODELS[r] for r in ROLES}


def save_role_models(data: dict) -> dict:
    clean = {}
    for r in ROLES:
        m = data.get(r) or {"type": "local"}
        if m.get("type") == "api":
            clean[r] = {"type": "api", "base_url": (m.get("base_url") or "").rstrip("/"),
                        "api_key": m.get("api_key") or "", "model": m.get("model") or ""}
        else:
            clean[r] = {"type": "local"}
    config.set("agent_models", clean)
    return public_role_models()


def public_role_models() -> dict:
    """Mesma config, sem expor a chave inteira na tela."""
    out = {}
    for r, m in role_models().items():
        m = dict(m)
        if m.get("api_key"):
            m["api_key"] = "…" + m["api_key"][-4:]
            m["has_key"] = True
        out[r] = m
    return out


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


class Stopped(Exception):
    pass


class AgentRun:
    def __init__(self, workspace: str, task: str, max_rounds: int = 4) -> None:
        self.ws = Path(workspace).expanduser().resolve()
        self.task = task.strip()
        self.max_rounds = max(1, min(int(max_rounds), 8))
        self.events: list[dict] = []
        self.status = "rodando"
        self.stop_flag = False
        self.files: dict[str, str] = {}
        self.spec = ""
        self.plan_files: list[str] = ["main.py"]
        self.started = time.time()

    # ------------------------------------------------------------ utilidades
    def emit(self, role: str, kind: str, text: str = "", **extra) -> None:
        self.events.append({"i": len(self.events), "t": round(time.time() - self.started, 1),
                            "role": role, "kind": kind, "text": text, **extra})

    def _check(self) -> None:
        if self.stop_flag:
            raise Stopped()

    def _budget_chars(self) -> int:
        s = engine.settings or {}
        ctx = s.get("n_ctx") or 4096
        return max(4000, int((ctx - 1500) * 2.6))

    def ask(self, role: str, system: str, user: str, max_tokens: int = 3072) -> str:
        self._check()
        m = role_models()[role]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        t0 = time.time()
        if m["type"] == "api":
            if not m.get("base_url") or not m.get("model"):
                raise RuntimeError(f"Configure endereço e modelo da API para o papel {ROLES[role]}.")
            body = json.dumps({"model": m["model"], "messages": messages,
                               "max_tokens": max_tokens, "temperature": 0.2}).encode()
            headers = {"Content-Type": "application/json"}
            if m.get("api_key"):
                headers["Authorization"] = f"Bearer {m['api_key']}"
            req = urllib.request.Request(m["base_url"] + "/chat/completions", data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.loads(r.read())
        else:
            if not engine.status()["ready"]:
                raise RuntimeError("Nenhum modelo local pronto. Inicie um em Meus modelos "
                                   "ou configure uma API externa para este papel.")
            data = engine.chat(messages, max_tokens=max_tokens)
        self._check()
        text = _strip_thinking(data["choices"][0]["message"].get("content", ""))
        toks = (data.get("usage") or {}).get("completion_tokens")
        self.emit(role, "llm", "", seconds=round(time.time() - t0, 1), tokens=toks,
                  source=m.get("model") or "local")
        return text

    def _safe_path(self, rel: str) -> Path:
        rel = rel.strip().strip("/").replace("\\", "/")
        p = (self.ws / rel).resolve()
        if self.ws not in p.parents and p != self.ws:
            raise ValueError(f"caminho fora da pasta de trabalho: {rel}")
        if p.suffix not in (".py", ".txt", ".md", ".json", ".toml", ".cfg", ".ini", ".csv"):
            raise ValueError(f"tipo de arquivo não permitido: {rel}")
        return p

    def write_blocks(self, role: str, text: str, allow_tests: bool, allow_code: bool) -> list[str]:
        written = []
        blocks = FILE_RE.findall(text)
        if not blocks:
            # modelos pequenos às vezes esquecem o cabeçalho: usa o 1º bloco de código
            m = re.search(r"```(?:python|py)?\n(.*?)\n```", text, re.S)
            target = self._default_target(allow_tests)
            if m and target:
                blocks = [(target, m.group(1))]
        for rel, content in blocks:
            rel = rel.strip()
            is_test = rel.startswith("tests/") or Path(rel).name.startswith("test_")
            if (is_test and not allow_tests) or (not is_test and not allow_code):
                self.emit(role, "warn", f"Ignorado: {rel} (fora do papel)")
                continue
            try:
                p = self._safe_path(rel)
            except ValueError as e:
                self.emit(role, "warn", str(e))
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content.rstrip() + "\n", encoding="utf-8")
            self.files[rel] = content
            written.append(rel)
        if written:
            self.emit(role, "files", "", files=written)
        return written

    def _default_target(self, tests: bool) -> str | None:
        return "tests/test_main.py" if tests else (self.plan_files[0] if self.plan_files else "main.py")

    def context(self, include_tests: bool = True) -> str:
        parts, used = [], 0
        budget = self._budget_chars()
        for rel in sorted(self.files):
            if not include_tests and rel.startswith("tests/"):
                continue
            block = f"ARQUIVO: {rel}\n```python\n{self.files[rel]}\n```\n"
            if used + len(block) > budget:
                parts.append(f"(arquivo {rel} omitido por falta de contexto)")
                continue
            parts.append(block)
            used += len(block)
        return "\n".join(parts)

    def run_tests(self) -> tuple[bool, str]:
        self._check()
        has_pytest = importlib.util.find_spec("pytest") is not None
        cmd = ([sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider", "tests"]
               if has_pytest else [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."])
        env = {k: v for k, v in os.environ.items() if not k.upper().endswith(("TOKEN", "KEY", "SECRET"))}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(self.ws)
        (self.ws / "tests").mkdir(exist_ok=True)
        init = self.ws / "tests" / "__init__.py"
        if not init.exists():
            init.write_text("")
        try:
            r = subprocess.run(cmd, cwd=self.ws, capture_output=True, text=True, timeout=90, env=env)
            out = (r.stdout + "\n" + r.stderr).strip()
            ok = r.returncode == 0
        except subprocess.TimeoutExpired:
            ok, out = False, "Os testes passaram de 90 segundos (provável loop infinito)."
        return ok, out[-3500:]

    # ---------------------------------------------------------------- fluxo
    def run(self) -> None:
        try:
            self.ws.mkdir(parents=True, exist_ok=True)
            for f in self.ws.rglob("*.py"):  # aproveita o que já existe na pasta
                if "__pycache__" not in f.parts and ".venv" not in f.parts and f.stat().st_size < 60_000:
                    self.files[str(f.relative_to(self.ws)).replace("\\", "/")] = f.read_text(errors="replace")
            self.plan()
            self.write_tests()
            self.develop()
            self.loop()
        except Stopped:
            self.status = "parado"
            self.emit("sistema", "done", "Parado por você.")
        except Exception as e:  # noqa: BLE001
            self.status = "erro"
            self.emit("sistema", "error", str(e))

    def plan(self) -> None:
        self.emit("arquiteto", "start", "Lendo o pedido e montando a especificação…")
        existing = ", ".join(sorted(self.files)) or "nenhum"
        text = self.ask("arquiteto",
            "Você é um arquiteto de software. Projeta soluções pequenas, simples e testáveis em Python "
            "(só biblioteca padrão, a menos que o pedido exija outra coisa).",
            f"Pedido do usuário:\n{self.task}\n\nArquivos já existentes: {existing}\n\n"
            "Responda SOMENTE com um JSON neste formato:\n"
            '{"resumo": "o que será construído, em 1 a 2 frases",\n'
            ' "arquivos": ["caminho.py"],\n'
            ' "interface": "funções/classes públicas com assinatura e o que retornam",\n'
            ' "comportamentos": ["regra verificável 1", "regra verificável 2", "caso de erro"]}\n'
            "Use no máximo 3 arquivos de código (sem contar testes). Não escreva código.",
            max_tokens=1200)
        data = _loose_json(text)
        if data:
            self.plan_files = [f for f in data.get("arquivos", []) if f.endswith(".py") and not f.startswith("tests")][:3]
            beh = "\n".join(f"- {b}" for b in data.get("comportamentos", []))
            self.spec = (f"Resumo: {data.get('resumo', '')}\nArquivos: {', '.join(self.plan_files)}\n"
                         f"Interface: {data.get('interface', '')}\nComportamentos:\n{beh}")
        else:
            self.plan_files = ["main.py"]
            self.spec = text[:3000]
        self.emit("arquiteto", "plan", self.spec)

    def write_tests(self) -> None:
        self._check()
        self.emit("testador", "start", "Escrevendo os testes antes do código…")
        text = self.ask("testador",
            "Você é um engenheiro de testes. Escreve testes com unittest (compatíveis com pytest), "
            "claros e independentes. Testa o comportamento descrito, sem inventar requisitos.",
            f"Especificação:\n{self.spec}\n\n"
            "Escreva os testes em tests/test_<nome>.py importando os módulos pelos nomes da especificação "
            "(ex.: from calculadora import somar). Cubra cada comportamento e pelo menos um caso de erro. "
            "No máximo 12 testes.\n\n" + FORMAT,
            max_tokens=2500)
        if not self.write_blocks("testador", text, allow_tests=True, allow_code=False):
            raise RuntimeError("O testador não devolveu nenhum arquivo de teste no formato esperado. "
                               "Tente de novo ou use um modelo maior para esse papel.")

    def develop(self) -> None:
        self._check()
        self.emit("desenvolvedor", "start", "Escrevendo o código para passar nos testes…")
        text = self.ask("desenvolvedor", _DEV_SYSTEM,
            f"Especificação:\n{self.spec}\n\nTestes que o código precisa passar:\n"
            f"{self._tests_only()}\n\nEscreva os arquivos: {', '.join(self.plan_files)}.\n\n" + FORMAT)
        if not self.write_blocks("desenvolvedor", text, allow_tests=False, allow_code=True):
            raise RuntimeError("O desenvolvedor não devolveu código no formato esperado.")

    def loop(self) -> None:
        contested = False
        for rnd in range(1, self.max_rounds + 1):
            self.emit("executor", "start", f"Rodando os testes (rodada {rnd} de {self.max_rounds})…")
            ok, out = self.run_tests()
            summary = _summarize(out)
            self.emit("executor", "tests", out, ok=ok, round=rnd, summary=summary)
            if ok:
                self.status = "concluído"
                self.emit("sistema", "done", f"Todos os testes passaram na rodada {rnd}.",
                          files=sorted(self.files))
                return
            if rnd == self.max_rounds:
                break
            self.emit("desenvolvedor", "start", "Lendo o erro e corrigindo…")
            text = self.ask("desenvolvedor", _DEV_SYSTEM,
                f"Especificação:\n{self.spec}\n\nArquivos atuais:\n{self.context()}\n\n"
                f"Saída dos testes (falhou):\n```\n{out[-2500:]}\n```\n\n"
                "Corrija o CÓDIGO (não os testes). Reescreva só os arquivos que mudarem.\n"
                "Se você tiver certeza de que um teste contradiz a especificação, responda apenas com "
                "uma linha 'CONTESTAR: <motivo>' em vez de código.\n\n" + FORMAT)
            if text.strip().upper().startswith("CONTESTAR") and not contested:
                contested = True
                reason = text.strip()[:600]
                self.emit("desenvolvedor", "msg", reason)
                self.emit("testador", "start", "Revisando os testes contestados…")
                fix = self.ask("testador",
                    "Você é um engenheiro de testes rigoroso. Só altera um teste se ele contradiz a "
                    "especificação; nunca para facilitar a passagem do código.",
                    f"Especificação:\n{self.spec}\n\nTestes atuais:\n{self._tests_only()}\n\n"
                    f"O desenvolvedor contestou: {reason}\n\nSaída dos testes:\n```\n{out[-1500:]}\n```\n\n"
                    "Se a contestação estiver correta, reescreva o arquivo de teste. Se não, responda "
                    "'MANTIDO: <motivo>'.\n\n" + FORMAT, max_tokens=2500)
                if fix.strip().upper().startswith("MANTIDO"):
                    self.emit("testador", "msg", fix.strip()[:600])
                else:
                    self.write_blocks("testador", fix, allow_tests=True, allow_code=False)
                continue
            if not self.write_blocks("desenvolvedor", text, allow_tests=False, allow_code=True):
                self.emit("desenvolvedor", "warn", "Nenhum arquivo reconhecido na correção.")
        self.status = "incompleto"
        self.emit("sistema", "done",
                  f"Os testes ainda falham depois de {self.max_rounds} rodadas. Os arquivos ficaram na pasta; "
                  "tente mais rodadas ou um modelo maior para o papel de Desenvolvedor.",
                  files=sorted(self.files))

    def _tests_only(self) -> str:
        return "\n".join(f"ARQUIVO: {r}\n```python\n{c}\n```" for r, c in sorted(self.files.items())
                         if r.startswith("tests/") and c.strip())


_DEV_SYSTEM = ("Você é um desenvolvedor Python cuidadoso. Escreve código simples, completo e "
               "funcional, só com a biblioteca padrão quando possível. Nunca altera os testes.")


def _loose_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    raw = m.group(0)
    for attempt in (raw, re.sub(r",\s*([}\]])", r"\1", raw)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    return None


def _summarize(out: str) -> str:
    for pat in (r"(\d+ (?:passed|failed)[^\n]*)", r"(Ran \d+ tests?[^\n]*\n+[^\n]*)"):
        m = re.findall(pat, out)
        if m:
            return m[-1].strip().replace("\n", " ")
    return out.strip().splitlines()[-1][:200] if out.strip() else ""


class AgentManager:
    def __init__(self) -> None:
        self.current: AgentRun | None = None

    def start(self, workspace: str, task: str, max_rounds: int) -> dict:
        if self.current and self.current.status == "rodando":
            raise RuntimeError("Já existe uma tarefa rodando. Pare a atual primeiro.")
        if not workspace.strip():
            raise ValueError("Escolha uma pasta de trabalho.")
        if len(task.strip()) < 10:
            raise ValueError("Descreva o que você quer construir com um pouco mais de detalhe.")
        ws = Path(workspace).expanduser().resolve()
        home = Path.home().resolve()
        if ws == home or ws == Path(ws.anchor) or len(ws.parts) < 3:
            raise ValueError("Use uma pasta dedicada ao projeto, não a pasta pessoal ou a raiz do disco.")
        self.current = AgentRun(str(ws), task, max_rounds)
        threading.Thread(target=self.current.run, daemon=True).start()
        config.set("agent_last_workspace", str(ws))
        return self.state(0)

    def stop(self) -> None:
        if self.current:
            self.current.stop_flag = True

    def state(self, since: int = 0) -> dict:
        r = self.current
        if not r:
            return {"status": "parado", "events": [], "last_workspace": config.get("agent_last_workspace")}
        return {"status": r.status, "workspace": str(r.ws), "task": r.task,
                "events": r.events[since:], "files": sorted(r.files)}

    def read_file(self, rel: str) -> str:
        if not self.current:
            raise ValueError("Nenhuma tarefa.")
        return self.current._safe_path(rel).read_text(encoding="utf-8", errors="replace")


agent = AgentManager()
