# Changelog

Mudanças de cada versão do Farol.

## v0.3.3 — 2026-10-26

### Corrigido
- "Não consegui instalar um motor pronto (versão v0.7.0)". O llama.cpp passou a publicar versões numeradas (v0.x) sem programas prontos. Os arquivos para baixar ficam nas compilações frequentes (bNNNN). Agora o Farol percorre os lançamentos recentes, do mais novo para o mais antigo, e usa o primeiro que tenha o arquivo certo para a máquina.
- A escolha do arquivo aceita variações no nome (por exemplo, `cuda-12.4` ou `cuda12.4`, `x64` ou `x86_64`, `.zip` ou `.tar.gz`).

## v0.3.2 — 2026-10-26

### Corrigido
- A tela mostrava "32/32 camadas na GPU" mesmo quando o motor instalado era a versão só CPU do llama-cpp-python, que ignora a placa de vídeo. Agora o Farol verifica se o motor usa a GPU de verdade (`llama_supports_gpu_offload` no llama-cpp-python, `--list-devices` no llama-server) e ajusta as camadas e o aviso de acordo.
- Com o motor só CPU, o ajuste usava apenas 4 threads, pensando que a GPU faria o trabalho. Agora usa todos os núcleos físicos.

### Adicionado
- Quando a máquina tem placa de vídeo e o motor instalado é só CPU, a aba Meus modelos sugere instalar o motor oficial do llama.cpp.

## v0.3.1 — 2026-10-26

### Corrigido
- Conversa muito mais lenta que o teste de velocidade (por exemplo, 130 tok/s no teste e 8 tok/s no chat). No mesmo computador, o chat agora recebe a resposta direto do motor. O servidor do Farol também passou a enviar cada pedaço na hora, sem esperar o TCP juntar pacotes (algoritmo de Nagle), o que no Windows atrasava o streaming.

### Adicionado
- O chat mostra duas velocidades: a do motor (medida pelo próprio llama-server) e a que chega na tela. Se a da tela for bem menor, ele avisa que a entrega está atrasando.
- Modelos que "pensam" antes de responder (que mandam o raciocínio separado, em `reasoning_content`) agora mostram esse raciocínio em cinza acima da resposta, em vez de parecerem travados.

## v0.3 — 2026-10-26

### Adicionado
- Instalação do motor oficial do llama.cpp (`llama-server`) pelo próprio app, baixado do GitHub na versão certa para a máquina: CUDA para NVIDIA (com as DLLs no Windows), Vulkan para AMD e Intel, Metal no Mac, ou CPU.
- Plano B na instalação: se a versão de GPU não abrir, o app instala a versão de CPU sozinho.
- Recuperação automática quando o motor cai ao carregar o modelo. O app tenta de novo com configurações cada vez mais seguras (sem flash attention, depois só CPU, depois contexto mínimo) e mostra as tentativas na tela.
- Escolha entre dois motores na aba Meus modelos: o oficial do llama.cpp (recomendado) e o llama-cpp-python.

### Corrigido
- Erro `access violation` ao iniciar modelos com o llama-cpp-python no Windows. Agora o app oferece trocar para o motor oficial e tenta configurações mais seguras antes de desistir.
- A verificação de "modelo pronto" passou a usar o `/health` do llama-server, que só responde OK quando o modelo terminou de carregar.

## v0.2 — 2026-10-26

### Adicionado
- Instalação do motor com um clique: `llama-cpp-python[server]` num ambiente separado (`~/.farol/venv`), já na versão pronta para NVIDIA, Apple ou CPU.
- Ajuste automático de desempenho: o app lê os metadados do arquivo GGUF (camadas, contexto máximo, cabeças de atenção) e escolhe camadas na GPU, tamanho do contexto, threads e flash attention. Também dá para ajustar à mão e reiniciar.
- Aba Conversar, com resposta em tempo real e tokens/s. Ela abre sozinha quando o modelo termina de carregar.
- Aba Agente, de programação: arquiteto → testador (escreve os testes antes do código) → desenvolvedor → execução real dos testes → correção em ciclo. Cada papel pode usar o modelo local ou uma API externa compatível com OpenAI.
- Travas de segurança do agente: só escreve na pasta de trabalho, só executa os testes, tem limite de 90 segundos e não repassa variáveis de ambiente com segredos.

### Corrigido
- Caminhos de modelos com barra invertida (Windows) quebravam os botões Iniciar e Apagar.

## v0.1 — 2026-10-26

### Adicionado
- Interface web local, feita só com a biblioteca padrão do Python (`python farol.py`).
- Login no Hugging Face por token, guardado só no computador (`~/.farol/config.json`, com permissão restrita).
- Detecção de hardware (CPU, RAM, GPU NVIDIA, AMD e Apple Silicon, disco) e teste de desempenho da memória e do processador.
- Farol por modelo (verde, amarelo ou vermelho), com a explicação de onde roda, a memória necessária, a velocidade prevista e uma dica.
- Escada de tamanhos (1B a 70B) mostrando o que a máquina aguenta.
- Busca de modelos GGUF no Hugging Face, com a versão recomendada para a máquina destacada.
- Download com barra de progresso, retomada e suporte a modelos divididos em várias partes.
- Teste de velocidade real com o modelo rodando, para comparar com a previsão.
- API compatível com a da OpenAI em `http://localhost:8080/v1`, com streaming.
