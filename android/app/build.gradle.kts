plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "br.org.ceia.cemigpoc"
    compileSdk = 34

    defaultConfig {
        applicationId = "br.org.ceia.cemigpoc"
        minSdk = 31
        targetSdk = 34
        versionCode = 2
        versionName = "1.1.0"

        ndk {
            abiFilters.add("arm64-v8a")
        }

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }

        // --- Seleção do sintetizador por build (task poc-engine-upgrade) ---
        // Entrega o SUPORTE ao 2.6B thinking-OFF sem trocar o default do app. O 2.6B é o
        // sintetizador vencedor (7.1x gate do 1.2B sobre a v3), mas seu chat_template força
        // <think>; o engine agora suprime o raciocínio (ver LlamaCppEngine.suppressReasoning /
        // llama_jni.cpp). A troca definitiva ocorre quando o treino DPO fechar.
        //   ./gradlew assembleRelease -Pcemig.synthModel=2.6b   -> LFM2.5-2.6B-Q4_0 + thinking-OFF
        //   (default / omitido)                                 -> LFM2.5-1.2B-Instruct-QAD-Q4_0
        // toolModel: escolhe o modelo TOOL-CALLING do pipeline híbrido (task poc-tools-2.6b).
        // O capitão quer DUAS opções comparáveis de embarque, selecionáveis por build:
        //   (default / omitido)            -> 1.2B tool r128 Q4 (latência de voz ~6-8s; embarcado)
        //   -Pcemig.toolModel=2.6b         -> 2.6B tool r64 Q4 (qualidade maior, mas ~18-23s/4.3GB)
        // O 2.6B é reasoning: SUPPRESS_REASONING=true (thinking-OFF via llama_jni). O GGUF do 2.6B
        // tool entra por override externo sob o nome LFM2.5-2.6B-tools-Q4_0.gguf.
        // task lfm3-f7: DEFAULT = LFM3-1.7B f4-r32-ep2 Q4_0 (ponto de operação da F6: XML
        // <tool_call>, manuais + compressão). -Pcemig.toolModel=1.2b volta ao LFM2.5 1.2B tool.
        val toolModel = (project.findProperty("cemig.toolModel") as String?)?.lowercase() ?: "lfm3"
        val useLfm3 = toolModel == "lfm3"
        val useTool26b = toolModel == "2.6b"
        buildConfigField("boolean", "USE_LFM3", useLfm3.toString())
        // synthModel (legado, task poc-engine-upgrade): sintetizador NÃO-tool (pipeline fixo antigo).
        val synthModel = (project.findProperty("cemig.synthModel") as String?)?.lowercase() ?: "1.2b"
        val use26b = synthModel == "2.6b" || useTool26b
        // task lfm3-q4km-test: quant do LFM3 por build. DEFAULT = Q4_0 (produção, decisão de latência
        // do capitão): o Q4_K_M retém mais qualidade (−2,3 vs −4,9 p.p. vs bf16) mas o prefill no S24+
        // cai 25% → +1,1 s no 1º token em toda pergunta (bench/lfm3_q4km/RESULTS.md).
        //   -Pcemig.lfm3Quant=q4_k_m  -> opção qualidade (GGUF Q4_K_M por override externo)
        val lfm3Quant = when ((project.findProperty("cemig.lfm3Quant") as String?)?.lowercase() ?: "q4_0") {
            "q4_0" -> "Q4_0"
            "q4_k_m" -> "Q4_K_M"
            else -> error("cemig.lfm3Quant deve ser q4_0 ou q4_k_m")
        }
        buildConfigField("String", "LFM3_QUANT", "\"$lfm3Quant\"")
        // task lfm3-f9-promote: checkpoint do LFM3 por build. DEFAULT = v3s400 (P2 RLVR v3, step 400,
        // LoRA sobre o f4-r32-ep2 mesclado; bateria ×1536: distrator aluc. 23,8→15,7, e2e app 64,1→71,8,
        // POC 151 12,6→21,2 — bench/lfm3_p1p2/RESULTS.md, bench/lfm3_f9/RESULTS.md). Mesmo template,
        // mesmo system e mesma arquitetura do f4: só o GGUF muda (override externo).
        //   -Pcemig.lfm3Model=f4  -> volta ao f4-r32-ep2 (APK v1.0)
        val lfm3Model = when ((project.findProperty("cemig.lfm3Model") as String?)?.lowercase() ?: "v3s400") {
            "v3s400" -> "v3s400"
            "f4" -> "f4-r32-ep2"
            else -> error("cemig.lfm3Model deve ser v3s400 ou f4")
        }
        if (lfm3Model == "v3s400" && lfm3Quant != "Q4_0") error("v3s400 só tem GGUF Q4_0 (produção)")
        buildConfigField("String", "LFM3_MODEL", "\"$lfm3Model\"")
        val llmModelName = when {
            useLfm3 -> "lfm3-1.7b-$lfm3Model-$lfm3Quant.gguf"
            useTool26b -> "LFM2.5-2.6B-tools-Q4_0.gguf"
            synthModel == "2.6b" -> "LFM2.5-2.6B-Q4_0.gguf"
            else -> "LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf"
        }
        buildConfigField("String", "LLM_MODEL_NAME", "\"$llmModelName\"")
        // O 2.6B é modelo de raciocínio (template prima <think>): thinking-OFF obrigatório.
        // No 1.2B (não-reasoning) a supressão é no-op seguro, mas só ativamos no 2.6B.
        buildConfigField("boolean", "SUPPRESS_REASONING", use26b.toString())

        // --- Seleção do motor ASR por build (task poc-sft-v3, Parte 6) ---
        // O capitão quer experimentar o Nemotron 3.5 INT8 (sherpa-onnx) no lugar do Whisper
        // Base, "mesmo que fique bem apertado em RAM" — o Whisper não captura termo técnico
        // ("disjuntor" não sai); o Nemotron acerta 91,9% dos termos (asr/README.md).
        //   ./gradlew assembleRelease -Pcemig.asrEngine=nemotron  -> Nemotron 3.5 INT8 (sherpa)
        //   (default / omitido)                                   -> Whisper Base Q5_1
        // task lfm3-f7: default Nemotron (a melhor versão embarcada; Whisper segue como fallback
        // automático se os ONNX externos não existirem).
        val asrEngine = (project.findProperty("cemig.asrEngine") as String?)?.lowercase() ?: "nemotron"
        val useNemotron = asrEngine == "nemotron"
        buildConfigField("String", "ASR_ENGINE", "\"$asrEngine\"")
        buildConfigField("boolean", "USE_NEMOTRON_ASR", useNemotron.toString())
    }

    signingConfigs {
        create("release") {
            storeFile = file("poc-release.jks")
            storePassword = "cemigpoc123"
            keyAlias = "poc"
            keyPassword = "cemigpoc123"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("release")
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            applicationIdSuffix = ".debug"
            isDebuggable = true
            signingConfig = signingConfigs.getByName("release")
        }
    }

    aaptOptions {
        noCompress.addAll(listOf("gguf", "bin", "db", "f16"))
    }

    testOptions {
        unitTests {
            isReturnDefaultValues = true
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.11"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    implementation(project(":llama"))
    implementation(project(":whisper"))
    implementation(project(":sherpa"))

    val composeBom = platform("androidx.compose:compose-bom:2024.04.01")
    implementation(composeBom)
    androidTestImplementation(composeBom)

    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")
    implementation("androidx.activity:activity-compose:1.8.2")

    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0")

    // SQLite com FTS5 embutido (o SQLite do sistema Android NAO tem FTS5).
    // Fork mantido do requery/sqlite-android (SQLite 3.50); pacote io.requery.android.database.sqlite.
    implementation("mil.nga:sqlite-android:3500400")

    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")

    testImplementation("junit:junit:4.13.2")
    // org.json real p/ ler fixtures de paridade na JVM (o android.jar só traz stubs).
    testImplementation("org.json:json:20231013")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.8.0")
}
