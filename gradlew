#!/usr/bin/env bash
# Encaminhador conveniente para executar comandos Gradle na raiz do repositório
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/android/gradlew" -p "$DIR/android" "$@"
