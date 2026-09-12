# Como gravar as amostras de VOZ REAL para a bancada de WER

**Objetivo:** medir o WER do Whisper Base Q5_1 no S24+ com **fala humana natural** (a medição
histórica de 9,5% usou áudio sintético TTS, que não vale para campo). O capitão relatou que a
transcrição só acerta com fala lenta e pausada — queremos justamente capturar a fala natural.

Quando os áudios chegarem, a re-medição é **só rodar**:
```bash
python3 asr/scripts/bench_real_voice.py            # Base, config antiga vs nova
python3 asr/scripts/bench_real_voice.py --model small   # opcional: comparar Small
```

---

## Formato do áudio (obrigatório)
- **WAV PCM 16-bit, 16 kHz, MONO.** (É o formato que o app captura e o Whisper espera.)
- Se gravar em outro formato (celular costuma gravar m4a/aac 44,1 kHz estéreo), converta:
  ```bash
  ffmpeg -i entrada.m4a -ar 16000 -ac 1 -c:a pcm_s16le saida.wav
  ```
- O script avisa se algum arquivo não estiver em 16 kHz mono 16-bit.

## Quantas / duração
- **12 a 15 enunciados** (o mínimo estatístico decente; mais é melhor).
- **5 a 15 segundos** cada (é a janela típica de push-to-talk).
- **Fale NATURALMENTE**: ritmo normal, sem soletrar, com as hesitações e pausas de sempre.
  NÃO grave lento e pausado — o ponto é medir a fala real de campo.

## Com e sem ruído (2 tomadas dos mesmos enunciados, se possível)
Para separar o efeito do ruído do efeito da fala:
1. **Limpo:** em ambiente silencioso (sala fechada).
2. **Ruído de campo:** com ruído ambiente real (rua movimentada, motor/gerador ligado,
   vento, cabine de veículo com janela aberta). NÃO precisa ser 10 dB exato — ruído real basta.
   Nomeie com sufixo `_noisy` para diferenciar (ex.: `q01_noisy.wav`).

## Sotaque e falantes
- Se der, **2 ou 3 falantes diferentes** (inclui o sotaque mineiro de campo do público-alvo).
- Homens e mulheres, se possível.

---

## Passo a passo
1. Grave os enunciados abaixo (ou fale suas próprias dúvidas técnicas — o que importa é ter o
   texto de referência exato do que foi falado).
2. Salve os `.wav` (16 kHz mono) em **`asr/data/real_voice/audio/`**.
3. Preencha **`asr/data/real_voice/manifest.jsonl`** — uma linha JSON por áudio:
   ```json
   {"id": "q01", "file": "q01.wav", "ref": "Quais os procedimentos de desenergização da NR-10?"}
   ```
   - `ref` = transcrição EXATA do que foi dito (o gabarito para o WER).
   - Já deixei o manifesto pré-preenchido com os enunciados sugeridos abaixo — se você falar
     exatamente esses textos, só precisa gravar os arquivos com os nomes indicados.
4. Rode `python3 asr/scripts/bench_real_voice.py`.

## Enunciados sugeridos (mistura de perguntas de NR + fala coloquial de campo)
| id | arquivo | texto para falar |
| :-- | :-- | :-- |
| q01 | q01.wav | Quais são os procedimentos obrigatórios para a desenergização segundo a NR-10? |
| q02 | q02.wav | Qual a distância mínima de segurança pra trabalhar perto de uma rede de treze mil e oitocentos volts? |
| q03 | q03.wav | A luva isolante de borracha classe dois aguenta qual tensão máxima? |
| q04 | q04.wav | Onde é que eu instalo o aterramento temporário na chave seccionadora? |
| q05 | q05.wav | É obrigatório usar capacete com jugular e óculos de proteção? |
| q06 | q06.wav | Quais os requisitos da NR-35 pra ancoragem do cinto tipo paraquedista no poste? |
| q07 | q07.wav | Posso mexer no circuito energizado durante chuva forte? |
| q08 | q08.wav | Qual o equipamento de proteção coletiva pra abrir chave fusível? |
| q09 | q09.wav | O religador desarmou por sobrecorrente de fase na saída do alimentador. |
| q10 | q10.wav | Quais ferramentas isoladas a mil volts têm que estar na bolsa de lona? |
| q11 | q11.wav | Como faço o bloqueio e etiquetagem do religador da subestação? |
| q12 | q12.wav | Qual a sequência certa pra constatar ausência de tensão na rede aérea? |

> Dica: para as versões com ruído, repita os mesmos enunciados e salve como `q01_noisy.wav`,
> etc., adicionando as linhas correspondentes no manifesto (`"id": "q01_noisy"`).

---

## O que o harness mede
Para cada áudio, roda o Whisper Base Q5_1 no S24+ em **duas configs** e compara no MESMO áudio:
- **`old`** = réplica do JNI ANTIGO (sem temperature fallback, sem prompt de domínio).
- **`new`** = réplica do JNI NOVO/embarcado (fallback + gates + prompt pt-BR de domínio).

Saída: WER por áudio + agregado (WER médio, acurácia de termos, RTF/latência), salvo em
`asr/results/real_voice/real_voice_wer.json`. Assim isolamos o ganho real da nossa mudança
sobre voz humana — não sobre TTS.
```
```
