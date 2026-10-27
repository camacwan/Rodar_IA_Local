"""Estimativa de desempenho e farol (verde / amarelo / vermelho).

Regra de bolso usada pelo app: gerar um token exige ler o modelo inteiro da
memória uma vez. Então tokens/s ≈ banda_de_memória × eficiência ÷ tamanho_do_modelo.
É uma estimativa — o botão "Testar de verdade" mede o número real.
"""
from __future__ import annotations

import re

OS_RESERVE_GB = 2.5      # o sistema e o navegador também precisam de RAM
CPU_EFF = 0.55           # fração da banda que o llama.cpp aproveita na CPU
GPU_EFF = 0.60

QUANT_RE = re.compile(
    r"(IQ[1-4]_(?:XXS|XS|S|M|NL)|Q[2-8]_K_(?:XS|S|M|L)|Q[2-8]_K|Q[4-8]_[01]|MXFP4|BF16|F16|F32)",
    re.I,
)

QUANT_INFO = {
    "Q2": ("Muito comprimido", "Cabe em quase tudo, mas perde bastante qualidade."),
    "IQ1": ("Extremo", "Só para experimentar. Qualidade bem baixa."),
    "IQ2": ("Muito comprimido", "Bom para modelos grandes em máquinas pequenas; perde qualidade."),
    "IQ3": ("Comprimido", "Equilíbrio razoável quando a memória é curta."),
    "Q3": ("Comprimido", "Menor e mais rápido, com perda perceptível."),
    "IQ4": ("Equilibrado", "Quase a qualidade do Q4_K_M, um pouco menor."),
    "Q4": ("Equilibrado", "O ponto ideal para a maioria: boa qualidade e tamanho moderado."),
    "MXFP4": ("Equilibrado", "Formato 4 bits nativo de alguns modelos recentes."),
    "Q5": ("Alta qualidade", "Mais fiel ao original, ocupa ~20% a mais que o Q4."),
    "Q6": ("Quase original", "Diferença mínima para o modelo completo."),
    "Q8": ("Praticamente original", "Grande e mais lento; vale se sobrar memória."),
    "F16": ("Original", "Sem compressão. Exige muita memória."),
    "BF16": ("Original", "Sem compressão. Exige muita memória."),
    "F32": ("Original (32 bits)", "Raramente faz sentido para rodar localmente."),
}


def parse_quant(filename: str) -> str:
    m = QUANT_RE.search(filename)
    return m.group(1).upper() if m else "?"


def quant_info(q: str) -> tuple[str, str]:
    for key in sorted(QUANT_INFO, key=len, reverse=True):
        if q.startswith(key):
            return QUANT_INFO[key]
    return ("Desconhecido", "Formato não identificado pelo nome do arquivo.")


def required_gb(size_gb: float, ctx: int = 4096) -> float:
    """Arquivo + cache de contexto (KV) + buffers do runtime."""
    kv = size_gb * 0.04 * (ctx / 4096) + 0.3
    return size_gb + kv + 0.4


def _speech(tps: float) -> str:
    words = tps * 0.75
    if tps >= 25:
        return f"≈{words:.0f} palavras/s — responde mais rápido do que você lê."
    if tps >= 12:
        return f"≈{words:.0f} palavras/s — fluido, no ritmo de leitura."
    if tps >= 4:
        return f"≈{words:.0f} palavras/s — dá para usar, mas você vai esperar um pouco."
    return f"≈{words:.1f} palavras/s — lento demais para conversar."


def estimate(size_gb: float, hw: dict, bench: dict | None, ctx: int = 4096) -> dict:
    need = required_gb(size_gb, ctx)
    ram_usable = max(0.0, hw["ram_total_gb"] - OS_RESERVE_GB)
    cpu_bw = (bench or {}).get("cpu_bandwidth_gbs") or 20.0  # palpite se ainda não mediu
    gpu = max(hw.get("gpus") or [], key=lambda g: g.get("vram_gb") or 0, default=None)

    mode, tps = "cpu", 0.0
    gpu_part = 0.0
    if gpu and gpu.get("vram_gb"):
        vram = gpu["vram_gb"] if gpu.get("unified") else gpu["vram_gb"] - 0.5
        gbw = gpu["bandwidth_gbs"] * GPU_EFF
        if need <= vram:
            mode, gpu_part = "gpu", 1.0
            tps = gbw / size_gb
        elif not gpu.get("unified") and vram > 1.5 and need <= vram + ram_usable:
            gpu_part = max(0.0, (vram - 0.8) / need)
            mode = "hibrido"
            t_tok = (size_gb * gpu_part) / gbw + (size_gb * (1 - gpu_part)) / (cpu_bw * CPU_EFF)
            tps = 1 / t_tok
    if mode == "cpu":
        tps = cpu_bw * CPU_EFF / size_gb

    if gpu and gpu.get("unified"):
        capacity = gpu["vram_gb"]
    elif mode in ("gpu", "hibrido"):
        capacity = (gpu["vram_gb"] - 0.5) + (ram_usable if mode == "hibrido" else 0)
    else:
        capacity = ram_usable
    fits = need <= capacity
    pressure = need / capacity if capacity else 9.9
    tps = min(tps, 250.0)  # acima disso o gargalo deixa de ser a memória
    # cabe "no aperto": só fechando outros programas (sem GPU dedicada)
    tight = not fits and mode == "cpu" and need <= capacity + 1.3

    if tight and tps >= 4:
        light = "amarelo"
        why = (f"Precisa de ~{need:.1f} GB e sobram ~{capacity:.1f} GB com o sistema aberto. "
               "Roda se você fechar navegador e outros programas pesados.")
        tip = "Se a máquina ficar lenta, use uma quantização um degrau menor."
    elif not fits:
        light = "vermelho"
        if mode == "cpu" and gpu is None and need <= hw["ram_total_gb"]:
            why = (f"Precisa de ~{need:.1f} GB e sobram ~{capacity:.1f} GB depois do sistema. "
                   "Até abriria fechando tudo, mas vai travar a máquina.")
        else:
            why = f"Precisa de ~{need:.1f} GB de memória e esta máquina oferece ~{capacity:.1f} GB."
        tip = "Tente uma quantização menor (Q3/IQ3) ou um modelo com menos parâmetros."
        tps = 0.0
    elif tps < 4:
        light = "vermelho"
        why = f"Cabe na memória, mas a estimativa é de só {tps:.1f} tokens/s."
        tip = "Prefira um modelo menor; este serve apenas para tarefas em lote, sem pressa."
    elif tps < 12 or pressure > 0.85:
        light = "amarelo"
        if pressure > 0.85:
            why = (f"Roda, mas ocupa {pressure*100:.0f}% da memória disponível — "
                   "feche outros programas antes.")
        else:
            why = f"Roda com cerca de {tps:.0f} tokens/s."
        tip = "Usável. Para mais velocidade, escolha uma quantização um degrau menor."
    else:
        light = "verde"
        why = f"Roda com folga: ~{tps:.0f} tokens/s e {need:.1f} GB de {capacity:.1f} GB."
        tip = "Boa escolha para esta máquina."

    where = {
        "gpu": f"Inteiro na GPU ({gpu['name']})" if gpu else "GPU",
        "hibrido": f"Dividido: {gpu_part*100:.0f}% na GPU, resto na RAM" if gpu else "",
        "cpu": "Só na CPU e RAM",
    }[mode]
    return {
        "light": light,
        "mode": mode,
        "where": where,
        "tokens_per_s": round(tps, 1),
        "required_gb": round(need, 1),
        "capacity_gb": round(capacity, 1),
        "memory_pressure": round(min(pressure, 9.9), 2),
        "why": why,
        "tip": tip,
        "feel": _speech(tps) if tps else "Não roda nesta máquina.",
        "estimated_with": "benchmark" if bench else "palpite (rode o teste de hardware)",
    }
