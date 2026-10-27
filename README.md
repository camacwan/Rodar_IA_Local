# Farol

Rode modelos de IA no seu computador sabendo, antes de baixar, se eles vão rodar bem.

- **Login no Hugging Face** por token, para baixar modelos com acesso restrito.
- **Teste de hardware**: mede a leitura de memória e o cálculo da CPU e detecta a GPU (NVIDIA, AMD, Apple Silicon).
- **Farol por modelo**: verde (roda bem), amarelo (roda com ressalvas) ou vermelho (não recomendado), com a explicação de onde roda, quanta memória usa e a velocidade prevista.
- **Download com retomada** e suporte a modelos divididos em várias partes.
- **Instalação do motor com um clique**: o app instala o `llama-cpp-python[server]` num ambiente separado (`~/.farol/venv`), já na versão pronta para NVIDIA (CUDA), Apple (Metal) ou só CPU, sem precisar compilar.
- **Ajuste automático**: lê o arquivo do modelo (camadas, contexto máximo, cabeças de atenção) e escolhe quantas camadas vão para a GPU, o tamanho do contexto, as threads e o flash attention. Dá para ajustar à mão e reiniciar.
- **Conversar**: ao iniciar um modelo, a aba de chat abre sozinha, com resposta em tempo real e tokens/s.
- **Agente de programação**: arquiteto → testador (escreve os testes antes) → desenvolvedor → execução real dos testes → correção em ciclo. Cada papel pode usar o modelo local ou uma API externa compatível com OpenAI.
- **Teste de velocidade real** depois de baixar, para comparar com a previsão.
- **API compatível com a da OpenAI** em `http://localhost:8080/v1`.

## Como rodar

Precisa de Python 3.10 ou mais novo. O app usa só a biblioteca padrão.

```bash
python farol.py
```

O navegador abre em `http://localhost:8080`.

Para **rodar** os modelos, escolha um motor na aba Meus modelos:

- **Motor oficial do llama.cpp** (recomendado). O app baixa o `llama-server` do GitHub na versão certa (CUDA para NVIDIA, Vulkan para AMD e Intel, Metal no Mac, ou CPU). Se a versão de GPU não abrir, ele instala a de CPU sozinho.
- **llama-cpp-python**. É um pacote Python, que pode ficar atrás do llama.cpp e não abrir modelos recentes.

Se o motor cair ao carregar o modelo, o Farol tenta de novo sozinho, cada vez com configurações mais seguras: sem flash attention, depois só CPU, depois contexto mínimo. As tentativas aparecem na tela.

Opcional, para medições mais precisas: `pip install psutil numpy`.

## Usar nos seus projetos

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
r = client.chat.completions.create(model="local", messages=[{"role": "user", "content": "Olá!"}])
print(r.choices[0].message.content)
```

Streaming (`stream=True`) também funciona.

## Como a previsão funciona

Para gerar cada token, o modelo inteiro é lido da memória uma vez. Então:

```
tokens/s ≈ banda de memória × eficiência ÷ tamanho do modelo
```

- **Banda**: medida no teste (CPU) ou tirada de uma tabela por modelo de placa (GPU / Apple).
- **Memória necessária**: arquivo + cache de contexto (4 mil tokens) + buffers.
- **Onde roda**: inteiro na GPU; dividido entre GPU e RAM; ou só na CPU.
- **Farol**: verde ≥ 12 tok/s com folga de memória; amarelo entre 4 e 12 tok/s, ou memória apertada; vermelho se não cabe ou se fica abaixo de 4 tok/s.

É uma estimativa (erra em torno de ±30%). O botão **Medir velocidade** (com o modelo rodando) mede o valor real.

## Agente: como funciona

1. **Arquiteto** transforma o pedido numa especificação curta (arquivos, interface, comportamentos).
2. **Testador** escreve testes `unittest` a partir da especificação, antes de existir código.
3. **Desenvolvedor** escreve o código. Ele não pode alterar os testes.
4. **Executor** (o próprio Python, não um modelo) roda os testes na pasta de trabalho.
5. Se falhar, o erro real volta para o desenvolvedor. Se ele contestar um teste, o testador revisa, e só altera se o teste contradizer a especificação.

Com modelo local, os três papéis usam o mesmo modelo, um de cada vez, então cabe na memória de máquinas modestas. Para máquinas fracas, a combinação que mais rende é usar o modelo local como arquiteto e testador e apontar só o desenvolvedor para uma API externa.

Segurança: o código gerado roda no seu computador. O agente só escreve dentro da pasta escolhida (e recusa a pasta pessoal e a raiz do disco), só executa os testes (nunca comandos livres), corta a execução em 90 segundos e não repassa variáveis de ambiente com TOKEN, KEY ou SECRET.

Nesta versão o agente gera projetos em Python. Funciona melhor com modelos de código de 7B ou mais (por exemplo, Qwen Coder).

## Onde ficam os arquivos

`~/.farol/` guarda `config.json` (token e chaves de API, com permissão só para o seu usuário), `models/` e `venv/` (o motor).
Para mudar o local, use a variável `FAROL_HOME`.

## Estrutura

```
farol.py                 inicia o app
farol_app/
  hardware.py            detecção e benchmark
  gguf.py                leitura dos metadados do modelo
  agent.py               agente de programação
  estimator.py           previsão de velocidade e farol
  hf.py                  login, busca e download no Hugging Face
  runtime.py             instala (binário ou pip), ajusta, inicia, recupera e para o motor
  server.py              API local + proxy /v1
  static/index.html      interface
```

## Próximos passos sugeridos

1. Agente para outras linguagens (JavaScript, SQL) e para editar projetos existentes do GitHub.
2. Empacotar como app de desktop (Tauri ou PyInstaller).
