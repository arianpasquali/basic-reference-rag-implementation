[x ]- Prepare repo
[x ]- initial import
[ ]- Data ingestion script
    [ ]- structured csv data into lancedb instance
    [ ]- unstructure data (pdfs) to vectors on lancedb instance

Refatorar codigo e limpar
[x]- desacoplar tools do agent
[x]- desacoplar agente da ui
[x]- deixar a ui com menos feature
[x]- remover arquivos que nao estao sendo usados
[x]- limpar settings e deixar mais simples
    [x]- fazer safety ser safe by default, no feature flag
[ ] incluir pre-commit for lint e code formatting
[x] manter no state os retrieved_documents and executed_sql_queries
[ ] remover ou simplificar makefile

Ingestao
[ ]- Fazer ser simples de seguir
[ ]- Testar com PDF maiores
[ ]- usar contextual retrieval

Router
[x] - supportar block off topic
[x] - supportar ask for more info

Retrieval
[ ]- Make sure we are using hybrid search
    [ ] - we are not embeddings the documents. we are using full text search only.
    [ ] - process documents embeddings vectors
[ ]- Make sure we have a optimization query node

Segurança
[x] - supportar guardrail basico
[] - suportar jailbreak
    - suportar huggingface jailbreak detection model
[] - nao vazar dados do schema da base de dados
[] - nao vazer system prompt

Documentacao
[ ]- Criar github repo
[x]- Criar README
[ ]- Criar diagrama no Excalidraw


Evals
[ ]- criar dataset
[ ]- criar script de retrieval eval
[ ]- criar script de eval
   [ ] - usar langsmith? ou also no terminal mais simples
