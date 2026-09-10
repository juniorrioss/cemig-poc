package br.org.ceia.cemigpoc.domain.model

/**
 * Representa a intenção de chamada de ferramenta disparada pelo modelo SLM.
 *
 * @property toolName Nome da ferramenta (normalmente "retriever")
 * @property query Consulta reformulada pelo modelo com termos assertivos para o BM25
 */
data class ToolCall(
    val toolName: String = "retriever",
    val query: String
)
