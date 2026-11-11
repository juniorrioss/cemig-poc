# F10 — distribuição: APK v1.2 instala limpo (tela de setup no 1º uso)

## Bug
Um terceiro instalou o APK v1.1 do HF em outro aparelho e recebeu "Falha na decisão do modelo: Motor
Llama não está carregado", com erro citando `ggml-base-q5_1.bin`. O APK v1.1 (236 MB) não tinha:

| faltava no APK v1.1 | efeito na instalação limpa | no S24+ de dev |
|---|---|---|
| GGUF do LFM3 v3s400 (1,05 GB) | LLM não carrega → "Motor Llama não está carregado" | `adb push` |
| Nemotron 3.5 (4 arquivos, 682 MB) | ASR cai no fallback Whisper, que também não existe → erro `ggml-base-q5_1.bin` | `adb push` |
| EmbeddingGemma-300M (278 MB) | busca densa desligada (sem aviso) | `adb push` |
| `dense_text.bin` / `dense_exponly.bin` (13,5 MB) | **busca caía em BM25 sem avisar** (achado novo: estavam fora dos assets, gitignorados) | resto de `adb push` antigo |

O Whisper (`ggml-base-q5_1.bin`) só é motor principal com `-Pcemig.asrEngine=whisper`. No build
default ele é só o fallback do Nemotron, e não é distribuído.

## Correção (linha A, decisão do capitão)
- `ModelDownloader.kt`: manifesto dos pesos externos do build (URL HF pública + tamanho + SHA-256).
  Baixa para `filesDir` em `.part` com **retomada por HTTP Range**, 5 tentativas com backoff,
  **SHA-256 conferido** antes do rename atômico e checagem de espaço livre. As fontes são o GGUF do
  nosso repo e o EmbeddingGemma e o Nemotron dos repos upstream, fixados por hash. Não há espelho.
- `MainViewModel`/`MainScreen`: se faltar algum arquivo, mostra o cartão **Configuração inicial**
  (lista, espaço livre, barra de progresso, Tentar novamente). Quando o download termina, carrega
  os motores. Se o modelo de voz falhar, o app fica pronto por texto e avisa que a voz está
  indisponível. Uma pergunta feita com o motor descarregado não chega ao pipeline.
- `ModelFileManager`: o Nemotron é procurado no override externo **ou** em `filesDir`. Arquivo
  ausente gera `MissingModelException` com o nome do arquivo e o que fazer, e o app volta ao setup.
- `dense_*.bin` agora embutidos nos assets. São os do S24+ validado, iguais por sha256 ao dataset
  `juniorrios/cemig-nr-retrieval-assets`.
- Permissão `INTERNET`, usada só pelo setup. `versionCode 3`, `versionName 1.2.0`.

## Validação: instalação limpa no Galaxy S24+ (simulando o terceiro)
`adb uninstall` (apaga dados e `/sdcard/Android/data/<pkg>`) → `adb install` do APK v1.2 → abrir
→ tela de setup → **Baixar modelos**.

- Download de 2,01 GB em **3,6 min** (Wi-Fi). Os 6 arquivos com SHA-256 verificado (logcat
  `ModelDownloader: OK … sha256 verificado`). **Retomada testada**: force-stop no meio do GGUF,
  e ao reabrir o botão mostrou só o restante (1,53 GB), continuando de ~480 MB.
- Motores prontos: Nemotron (de `filesDir`) + v3s400 + denso (2202×768) + manuais/compressor.
- 3 perguntas-chave (runner F7, mesmo processo da sessão limpa; `data/device/`):

| caso | modo | resultado |
|---|---|---|
| `02bb2740` NR-10 prontuário | texto | ✅ "carga instalada superior a 75 kW", NR-10 10.2.4; total 9,2 s |
| `26d6644d` manual "conversa ao pé do poste" | texto | ✅ manual NR-10 ligado; "no local do serviço", "superior e trabalhador ou equipe"; 9,2 s |
| `8b8bd5e9` escada fixa vertical | **voz** | ✅ Nemotron transcreveu certo (1,9 s); resposta **idêntica** à da v1.1 validada (F9); 12,5 s |

PSS 2,49 GB após as perguntas. Screenshots em `screenshots/` (setup, download, pronto).
O 8/8 completo fica para depois, por decisão do capitão (validação compacta).

## Publicação
- HF `juniorrios/automobil-ultra-semnexo`: `apk/cemig-assistente-v1.2.apk` (249.566.486 bytes,
  sha256 `19c4e12e929cbe60bb52ea5d54e3f728299076df3583c9064b75dbe48e91639b`). O v1.1 foi removido.
  O card está atualizado com instalação sem PC, requisitos (~2,6 GB livres, download inicial de
  2,01 GB) e a alternativa por `adb`. Espelho do card: `release/README.md`.
- O APK "offline completo" foi adiado por decisão do capitão.

## Build
`./gradlew :app:assembleRelease` (defaults: LFM3 v3s400 Q4_0 + Nemotron). Num worktree novo é
preciso popular `android/llama.cpp` e `android/whisper.cpp` (submódulos), os `.so` do sherpa
(`android/sherpa/src/main/jniLibs/arm64-v8a/`, gitignorados) e `android/app/poc-release.jks`.
