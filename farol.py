#!/usr/bin/env python3
"""Farol — rode modelos de IA no seu computador.

Uso:
    python farol.py              # abre em http://localhost:8080
    python farol.py --port 9000  # outra porta
    python farol.py --no-browser
"""
import argparse
import threading
import webbrowser

from farol_app.runtime import engine
from farol_app.server import serve


def main() -> None:
    ap = argparse.ArgumentParser(description="Farol — gerenciador de modelos locais")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1",
                    help="use 0.0.0.0 só se quiser acessar de outros aparelhos da rede")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    httpd = serve(args.host, args.port)
    url = f"http://localhost:{args.port}"
    print(f"\n  Farol rodando em {url}")
    print(f"  API compatível com OpenAI: {url}/v1   (Ctrl+C para sair)\n")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        httpd.server_close()


if __name__ == "__main__":
    main()
