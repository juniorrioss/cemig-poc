plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

// Módulo :sherpa — motor ASR Nemotron 3.5 streaming via sherpa-onnx (ONNX Runtime).
// Diferente do :whisper (compila nativo via CMake), aqui usamos os .so PRÉ-COMPILADOS
// (libsherpa-onnx-jni.so + libonnxruntime.so, arm64-v8a) em jniLibs/ e a API Kotlin
// oficial em com.k2fsa.sherpa.onnx (copiada do SherpaOnnxAar, casada com o JNI).
android {
    namespace = "br.org.ceia.cemigpoc.sherpa"
    compileSdk = 34
    ndkVersion = "26.1.10909125"

    defaultConfig {
        minSdk = 31
        ndk {
            abiFilters.add("arm64-v8a")
        }
    }

    // Os .so já vêm prontos; nada de externalNativeBuild.
    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("src/main/jniLibs")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0")
    testImplementation("junit:junit:4.13.2")
}
