package br.org.ceia.cemigpoc.data.fake

import br.org.ceia.cemigpoc.domain.engine.Retriever
import br.org.ceia.cemigpoc.domain.model.Chunk

/**
 * Implementação mock do motor de busca com 3 chunks oficiais de exemplo (NR-10 e NR-35).
 */
class FakeRetriever(
    private val chunks: List<Chunk> = DEFAULT_MOCK_CHUNKS,
    private val returnEmpty: Boolean = false
) : Retriever {

    companion object {
        val DEFAULT_MOCK_CHUNKS = listOf(
            Chunk(
                id = 1,
                doc = "NR-10",
                section = "Item 10.4.1 - Procedimentos de Desenergização",
                content = "São consideradas desenergizadas as instalações elétricas liberadas para trabalho, mediante os seguintes procedimentos: seccionamento, impedimento de reenergização, constatação da ausência de tensão, instalação de aterramento temporário com equipotencialização dos condutores, proteção dos elementos energizados e sinalização de impedimento.",
                score = -1.25
            ),
            Chunk(
                id = 2,
                doc = "NR-10",
                section = "Item 10.2.8.2 - Medidas de Proteção Coletiva",
                content = "Em todos os serviços executados em instalações elétricas devem ser previstas e adotadas, prioritariamente, medidas de proteção coletiva compreendendo a desenergização elétrica e, na sua impossibilidade, o emprego de tensão de segurança.",
                score = -0.95
            ),
            Chunk(
                id = 3,
                doc = "NR-35",
                section = "Item 35.5.1 - Sistema de Proteção Contra Quedas",
                content = "É obrigatória a utilização de sistema de proteção contra quedas sempre que o trabalho em altura for realizado a mais de dois metros do nível inferior, onde haja risco de queda. Deve ser utilizado cinturão de segurança tipo paraquedista com talabarte duplo.",
                score = -0.80
            )
        )
    }

    override suspend fun search(query: String, topK: Int): List<Chunk> {
        if (returnEmpty) return emptyList()

        val normalizedQuery = query.lowercase().trim()
        if (normalizedQuery.contains("vazio") || normalizedQuery.contains("sem_resultado")) {
            return emptyList()
        }

        // Filtra por relevância simples das palavras-chave
        val matched = chunks.filter { chunk ->
            val text = "${chunk.doc} ${chunk.section} ${chunk.content}".lowercase()
            normalizedQuery.split(" ").any { term ->
                term.length > 2 && text.contains(term)
            }
        }

        return if (matched.isNotEmpty()) {
            matched.take(topK)
        } else {
            // Se a busca genérica não encontrar filtro específico, retorna o padrão até topK
            chunks.take(topK)
        }
    }
}
