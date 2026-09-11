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
        versionCode = 1
        versionName = "1.0.0"

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
        val synthModel = (project.findProperty("cemig.synthModel") as String?)?.lowercase() ?: "1.2b"
        val use26b = synthModel == "2.6b"
        val llmModelName = if (use26b) "LFM2.5-2.6B-Q4_0.gguf" else "LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf"
        buildConfigField("String", "LLM_MODEL_NAME", "\"$llmModelName\"")
        // O 2.6B é modelo de raciocínio (template prima <think>): thinking-OFF obrigatório.
        // No 1.2B (não-reasoning) a supressão é no-op seguro, mas só ativamos no 2.6B.
        buildConfigField("boolean", "SUPPRESS_REASONING", use26b.toString())
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
        noCompress.addAll(listOf("gguf", "bin", "db"))
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
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.8.0")
}
