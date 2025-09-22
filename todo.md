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
[x] incluir pre-commit for lint e code formatting
[x] manter no state os retrieved_documents and executed_sql_queries
[x] remover ou simplificar makefile
[x] rever system prompt and use just prompts.py
[ ] rever logging setup utils/logging_config.py
[x] rever o searchdocuments results to document list
[x] refatorar Tools.py


UI
[x] criar um ficheiro com os starters

Ingestao
[x]- Fazer ser simples de seguir
[x]- Testar com PDF maiores
[x]- usar contextual retrieval
[x]- rever usando chroma_db

Router
[x] - supportar block off topic
[x] - supportar ask for more info

Retrieval
[x] - revisar codigo retrieval
[x]- Make sure we are using hybrid search
    [ ] - we are not embeddings the documents. we are using full text search only.
    [x] - process documents embeddings vectors
[ ]- Make sure we have a optimization query node

Segurança
[x] - supportar guardrail basico
[] - suportar jailbreak
    - suportar huggingface jailbreak detection model
[] - nao vazar dados do schema da base de dados
[] - nao vaaer system prompt

Documentacao
[x]- Criar github repo
[x]- Criar README
[x]- Criar diagrama no Excalidraw
[x] - clear readme from chromadb details
[ ] - have a dedicated readme for data ingestion

Evals
[ ] - revisar evals. simplificar
[ ] - fazer um dataset simples com pergunta e resposta
[ ] - fazer um dataset simples para avaliar router
[ ] - fazer um dataset simples para avaliar tool call
[x]- criar dataset
[x]- criar script de retrieval eval
[x]- criar script de eval
   [ ] - usar langsmith? ou also no terminal mais simples


Deploy
[x] - Pesquisar commo fazer deploy
    [ ] - Docker init deve criar o banco, ingerir csvs and processar pdfs.
    - Deploy em algum outro serviço que hospedada Docker?
        - Conta Azure particular?
[ ] configurar render.com para docker deploy
[ ] permitir reference documents ser links que abrem no browser
    [ ] pra isso os pdfs devem estar acessiveis via http
