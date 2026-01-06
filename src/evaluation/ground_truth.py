"""
Ground Truth - Respuestas correctas extraídas directamente de los PDFs
Fuentes:
- Propuesta_Tecnica_Analitica_Avanzada.pdf (Belcorp)
- TDD - Learning Journey BCP v2 - 20250429.pdf
"""

GROUND_TRUTH = {
    "¿Cuál es el objetivo principal de la arquitectura propuesta para Belcorp?":
        "Construir una plataforma de analítica avanzada alineada a la arquitectura de datos de "
        "Belcorp, que permita: agilidad en el desarrollo y despliegue de modelos, control y "
        "trazabilidad total de los ciclos de vida, seguridad y gobernanza transversal, e "
        "integración con las aplicaciones del negocio.",

    "¿Cuál es el flujo completo del sistema Learning Journey desde que se envía el correo hasta que se recibe la respuesta?":
        "El flujo completo es: (1) Logic App escucha una nueva entrada en el buzón designado, "
        "(2) el adjunto Excel es validado por estructura y nombre, (3) se invoca Azure Function "
        "con el archivo como input, (4) por cada skill-nivel se genera el embedding usando "
        "text-embedding-ada-002, se hace búsqueda en Qdrant (top K=40) y keyword search en "
        "PostgreSQL, se aplican filtros (idioma, dificultad, estado activo), se computa un "
        "ranking híbrido (semántico, RRF, rating, bonus multi-skill), se aplica algoritmo "
        "greedy set-cover (máx. 4/2/1 recursos por capacidad), y se consulta YouTube para "
        "complementar los cursos, (5) se arma un archivo Excel estructurado por Role con hoja "
        "oculta Audit_Log, (6) el archivo generado se guarda en Azure Blob Storage con "
        "convención learning_journeys/YYYY-MM-DD/role_timestamp.xlsx, (7) Logic App responde "
        "al remitente con el archivo adjunto y estado del procesamiento.",

    "¿Cuál es la fórmula de ranking utilizada para puntuar los cursos en Learning Journey y qué factores considera?":
        "La fórmula de ranking es: score = (0.60 * cosine_similarity) + (0.15 * RRF_keyword_boost) + "
        "(0.15 * normalized_rating) + (0.10 * multi_skill_bonus). Donde cosine_similarity es la "
        "cercanía entre el embedding del skill y el del curso, RRF_keyword_boost es el factor que "
        "premia si el título o descripción coincide con keywords clave, normalized_rating es la "
        "calificación del curso normalizada de 1-5 a 0-1, y multi_skill_bonus es el extra que "
        "recibe un curso si cubre más de un skill del mapa. Los cursos se ordenan de mayor a menor "
        "score final.",

    "¿Qué componentes de Azure se utilizan en el proyecto Learning Journey BCP y cuál es la función de cada uno?":
        "Los componentes de Azure son: Azure Logic App (orquestador que escucha correo con adjunto, "
        "lanza procesamiento IA, y envía respuesta al usuario), Azure Function - Ingesta mensual "
        "(serverless que extrae, normaliza y actualiza catálogos desde Udemy y MS Learn, y genera "
        "embeddings), Azure Function - Course Matching (serverless que procesa mapas de carrera, "
        "consulta base vectorial y estructurada, y arma recomendación), Qdrant en Azure Container Apps "
        "(base vectorial que almacena embeddings de cursos y permite búsqueda ANN semántica), "
        "PostgreSQL Flexible Server (base estructurada que almacena metadata de los cursos: categorías, "
        "idioma, dificultad, rating), Azure Blob Storage (almacenamiento que guarda los archivos Excel "
        "de entrada y salida, versionados y trazables), Application Insights (observabilidad que "
        "registra logs, métricas y errores de todas las ejecuciones), Azure Key Vault (seguridad que "
        "guarda credenciales, claves API y configuraciones sensibles), y Azure OpenAI (modelo IA que "
        "genera embeddings por skill-nivel usando text-embedding-ada-002).",

    "¿Qué modelo de embeddings se utiliza en Learning Journey y cuántas dimensiones tiene?":
        "El modelo de embeddings utilizado es text-embedding-ada-002 de Azure OpenAI, con un "
        "embedding vector de aproximadamente 1536 dimensiones."
}
