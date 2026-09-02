# HR Policy RAG Assistant — Azure AI Search + LangGraph

A grounded **Retrieval-Augmented Generation (RAG)** system for answering HR policy questions across **UK and Ireland**, built with **Azure Document Intelligence, Azure AI Search, OpenAI embeddings, LangGraph, RAGAS, and Logfire**.

The project focuses on an important production-RAG problem: **an LLM should not simply answer a question because it can generate an answer — it should first determine whether the question is clear, retrieve the appropriate evidence, assess whether the evidence is sufficient, generate a grounded response with citations, and verify the result.**

---

## 🚀 What This Project Demonstrates

This project implements an end-to-end grounded RAG pipeline:

```text
                    HR Policy PDFs
                          │
                          ▼
              ┌───────────────────────┐
              │ Azure Document        │
              │ Intelligence          │
              │ prebuilt-layout       │
              └───────────┬───────────┘
                          │
                          ▼
                  Markdown content
                          │
                          ▼
              ┌───────────────────────┐
              │ Two-Stage Chunking    │
              │                       │
              │ 1. Markdown headers   │
              │ 2. Semantic chunking  │
              └───────────┬───────────┘
                          │
                          ▼
                    Text chunks
                          │
                    + embeddings
                          │
                          ▼
              ┌───────────────────────┐
              │ Azure AI Search       │
              │                       │
              │ BM25 + Vector Search  │
              │        ↓              │
              │ Hybrid Retrieval      │
              │        ↓              │
              │ Semantic Reranker     │
              └───────────┬───────────┘
                          │
                     Top evidence
                          │
                          ▼
              ┌───────────────────────┐
              │      LangGraph        │
              │                       │
              │ Understand            │
              │      ↓                │
              │ Retrieve              │
              │      ↓                │
              │ Grade                 │
              │      ↓                │
              │ Generate              │
              │      ↓                │
              │ Verify                │
              └───────────┬───────────┘
                          │
                          ▼
                 Grounded Answer
                   + Citations
                          │
                          ▼
              ┌───────────────────────┐
              │ RAGAS Evaluation      │
              │ + Logfire Tracing     │
              └───────────────────────┘
```

---

## 🎯 Problem Statement

HR policy questions often look simple but can become ambiguous when policies differ by jurisdiction.

For example:

> "How much notice do I need to give for booking leave?"

The correct answer depends on whether the employee is covered by the **UK or Ireland** policy.

A conventional RAG system might retrieve documents from both jurisdictions and allow the LLM to choose an answer.

This project deliberately avoids that behaviour.

Instead, the system:

1. Determines whether the question is sufficiently clear.
2. Extracts the jurisdiction when explicitly stated or clearly implied.
3. Asks a clarification question when the jurisdiction is required but missing.
4. Retrieves evidence using hybrid search.
5. Filters results by jurisdiction.
6. Grades whether the retrieved evidence is sufficient.
7. Refuses to answer when the evidence does not support the requested information.
8. Generates an answer grounded in retrieved policy extracts.
9. Attaches citations to the supporting source.
10. Verifies the generated answer.

This makes the system substantially more suitable for **high-trust enterprise RAG scenarios**.

---

# 🧠 Key Features

### 1. Document Intelligence-based extraction

PDF policy documents are processed using **Azure Document Intelligence** with the `prebuilt-layout` model.

The extracted content is requested in Markdown format so that document structure can be preserved.

The project maintains document-level metadata including:

* Source filename
* Page number
* Jurisdiction
* Effective date

The current document registry includes UK and Ireland policies covering annual leave, disciplinary procedures, remote/hybrid working, employment contracts, and working-time guidance.

---

### 2. Two-stage chunking

Instead of splitting documents using a simple fixed-size text splitter, the project uses a two-stage approach.

#### Stage 1 — Structure-aware splitting

Markdown headings are used to preserve document hierarchy:

```text
# Heading
## Section
### Subsection
```

The implementation uses:

```python
MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3")
    ]
)
```

This produces meaningful section-level chunks rather than arbitrary text fragments.

#### Stage 2 — Semantic chunking

Large sections are further split using `SemanticChunker`.

The project uses a percentile-based semantic breakpoint threshold:

```python
SemanticChunker(
    embeddings=embedding,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90
)
```

Sections below the configured character threshold are retained as-is, while larger sections are semantically split.

This preserves semantic coherence while preventing excessively large retrieval units.

---

# 🔎 Azure AI Search

The resulting chunks are indexed in **Azure AI Search**.

The index is:

```text
hr-policies-hybrid
```

The index contains fields for:

| Field            | Purpose                   |
| ---------------- | ------------------------- |
| `id`             | Unique chunk identifier   |
| `content`        | Searchable policy content |
| `section`        | Document section          |
| `source_file`    | Original PDF              |
| `page`           | Source page               |
| `jurisdiction`   | UK / Ireland filtering    |
| `effective_date` | Policy effective date     |
| `content_vector` | Embedding vector          |

The vector field is configured for **3072-dimensional embeddings**, matching the `text-embedding-3-large` embedding model used by the project.

---

# 🔀 Hybrid Retrieval

The retrieval layer combines:

### BM25 keyword search

```python
search_text=query
```

with:

### Vector search

```python
VectorizedQuery(
    vector=embedding.embed_query(query),
    k_nearest_neighbors=20,
    fields="content_vector"
)
```

The two retrieval signals are combined through Azure AI Search hybrid search.

The query also supports metadata filtering:

```python
filter=f"jurisdiction eq '{jurisdiction}'"
```

This is particularly important for avoiding cross-jurisdiction contamination.

For example:

```text
Question:
How many days annual leave?

Jurisdiction:
UK
```

will not return Ireland policy chunks simply because they are semantically similar.

---

# 🏆 Semantic Reranking

After hybrid retrieval, Azure AI Search's semantic ranker is enabled using:

```python
query_type="semantic"
semantic_configuration_name="sem-config"
```

The semantic configuration prioritises:

* `section` as the title field
* `content` as the content field

The system retrieves the top results and exposes the semantic reranker score:

```python
@search.reranker_score
```

A sanity check in the notebook demonstrates that the UK annual-leave query returns UK documents with high reranking scores and keeps Ireland documents out through jurisdiction filtering.

---

# 🤖 LangGraph RAG Workflow

The RAG application is implemented as a **stateful LangGraph workflow**.

The graph contains five nodes:

```text
START
  │
  ▼
Understand
  │
  ├── Clarify ──────────────► END
  │
  ▼
Retrieve
  │
  ▼
Grade
  │
  ├── Refuse ───────────────► END
  │
  ▼
Generate
  │
  ▼
Verify
  │
  ▼
 END
```

The workflow is compiled using LangGraph's `StateGraph`.

---

## 1. Understand

The first node analyses the user's question using structured output.

The `QueryAnalysis` model determines:

```python
class QueryAnalysis(BaseModel):
    clear: bool
    jurisdiction: Literal["UK", "Ireland"] | None
    search_query: str
    clarifying_question: str | None
```

The model is explicitly instructed **not to guess the jurisdiction**.

If the question is ambiguous, the workflow terminates with a clarification request.

Example:

```text
User:
How much notice do I need to give for booking leave?

Assistant:
Are you employed in the UK or Ireland?
```

The notebook demonstrates this behaviour directly.

---

## 2. Retrieve

Once the question is sufficiently clear, the workflow performs hybrid retrieval:

```text
User query
    │
    ├── BM25
    │
    └── Vector search
            │
            ▼
       Hybrid results
            │
            ▼
       Semantic ranking
```

The jurisdiction is applied as a metadata filter when available.

---

## 3. Grade

The retrieved evidence is passed to a second structured-output model.

The `SourceGrade` model evaluates:

```python
class SourceGrade(BaseModel):
    sufficient: bool
    missing_facts: str | None
```

The grading step deliberately uses a strict rule:

> Partial coverage is insufficient.

This prevents the generator from filling missing information from its own pretrained knowledge.

If the evidence is insufficient, the system returns a grounded refusal instead of hallucinating an answer.

---

## 4. Generate

If the evidence is considered sufficient, the system generates an answer using the retrieved policy extracts.

The generated answer includes source references such as:

```text
[S1]
```

These references are subsequently mapped back to the original:

* Source file
* Page
* Section

The notebook demonstrates a response such as:

```text
A full-time UK employee is entitled to a minimum of
5.6 weeks, or 28 days, of paid annual leave per leave
year; this may include the 8 English bank holidays. [S1]
```

The corresponding citation identifies the source PDF, page, and policy section.

---

## 5. Verify

The final stage verifies the generated response against the retrieved evidence.

The workflow therefore follows:

```text
Retrieve
   ↓
Grade evidence
   ↓
Generate answer
   ↓
Verify answer
```

This creates an additional quality-control layer rather than assuming that a successful LLM generation is automatically correct.

---

# 🛡️ Grounded Refusal

One of the most important design principles in this project is:

> **If the knowledge base cannot support the answer, don't invent one.**

For example, when asked:

```text
What is the statutory sick pay rate per week in the UK?
```

the system identifies that the available policy extracts do not contain the required SSP rate and returns a refusal explaining what information is missing.

This is particularly valuable for HR, legal, financial, and other enterprise use cases where unsupported answers can be more harmful than no answer.

---

# 📚 Citation-Aware Answers

Every generated answer can be linked back to its retrieved evidence.

A citation contains metadata such as:

```text
Tag:         S1
Source:      uk_annual_leave_policy.pdf
Page:        1
Section:     Annual Leave Policy (United Kingdom)
```

The notebook demonstrates the resulting citation structure using a Pandas DataFrame.

This improves:

* Traceability
* Auditability
* User trust
* Debugging
* Evaluation

---

# 📊 RAGAS Evaluation

The project includes a RAG evaluation stage using **RAGAS**.

The evaluation dataset contains representative HR questions and expected references.

Examples include:

* UK annual leave entitlement
* Ireland public holidays
* Sick leave during booked annual leave
* UK disciplinary warning duration

The project evaluates RAG quality using metrics including:

* Faithfulness
* Answer relevancy
* Context precision
* Context recall

The notebook defines a small gold dataset specifically for these scenarios.

This provides an evaluation loop beyond simply checking whether the application "looks like it works."

---

# 🔭 Observability with Logfire

The application integrates **Logfire** for observability.

OpenAI calls and HTTP requests are instrumented:

```python
logfire.instrument_openai()
logfire.instrument_httpx()
```

Individual workflow executions are also wrapped in Logfire spans:

```python
with logfire.span("query: clear"):
    ...
```

This makes it possible to inspect:

* LLM calls
* Embedding calls
* Retrieval latency
* Workflow execution
* Query-specific traces

The notebook records traces for the different query paths, including clear, vague, and out-of-scope questions.

---

# 📂 Repository Structure

```text
hr-policy-rag-azure-langgraph/
│
├── documents/
│   ├── uk_annual_leave_policy.pdf
│   ├── ireland_annual_leave_policy.pdf
│   ├── uk_disciplinary_policy.pdf
│   ├── ireland_remote_work_policy.pdf
│   ├── remote_hybrid_working_policy.pdf
│   ├── uk_employment_contract.pdf
│   └── uk_working_time_guidance.pdf
│
├── Experiement.ipynb
│
├── extracted_cache.json
│
├── requirement.txt
│
└── README.md
```

The repository currently contains the notebook, document corpus, extracted-document cache, dependency file, and README.

---

# ⚙️ Technology Stack

| Component                   | Technology                              |
| --------------------------- | --------------------------------------- |
| Language                    | Python 3.13                             |
| Document extraction         | Azure Document Intelligence             |
| LLM                         | OpenAI / Azure-hosted OpenAI deployment |
| Embeddings                  | `text-embedding-3-large`                |
| Vector database / retrieval | Azure AI Search                         |
| Keyword retrieval           | BM25                                    |
| Vector retrieval            | HNSW                                    |
| Reranking                   | Azure AI Search Semantic Ranker         |
| Chunking                    | Markdown Header + Semantic Chunking     |
| Orchestration               | LangGraph                               |
| RAG framework               | LangChain                               |
| Evaluation                  | RAGAS                                   |
| Observability               | Logfire                                 |
| Data validation             | Pydantic                                |
| Notebook                    | Jupyter                                 |

The pinned environment includes LangChain, LangGraph, RAGAS, Azure Document Intelligence, Azure Search Documents, OpenAI, Logfire, Pydantic, Pandas, and related dependencies.

---

# 🔐 Configuration

Create a `.env` file in the project root.

Example:

```env
# Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://<resource>.cognitiveservices.azure.com/"
AZURE_DOCUMENT_INTELLIGENCE_KEY="<key>"

# Azure AI Search
AZURE_AI_SEARCH_ENDPOINT="https://<service>.search.windows.net"
AZURE_AI_SEARCH_KEY="<admin-key>"
INDEX_NAME="hr-policies-hybrid"

# OpenAI / Azure-hosted OpenAI configuration
OPENAI_API_KEY="<key>"
OPENAI_COMPLETION_DEPLOYMENT="<chat-deployment>"
OPENAI_EMBEDDING_DEPLOYMENT="<embedding-deployment>"

# Optional observability
LOGFIRE_TOKEN="<token>"

# Optional LangSmith tracing
LANGSMITH_TRACING="true"
LANGSMITH_PROJECT="<project>"
LANGSMITH_API_KEY="<key>"
```

**Never commit `.env` or API keys to Git.**

Add the following to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
.ipynb_checkpoints/
```

---

# 🧰 Installation

Clone the repository:

```bash
git clone https://github.com/nmanuvenugopal/hr-policy-rag-azure-langgraph.git

cd hr-policy-rag-azure-langgraph
```

Create a virtual environment:

```bash
python3.13 -m venv .venv
```

Activate it:

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirement.txt
```

The repository provides pinned versions for the core LangChain/LangGraph/RAGAS environment to reduce dependency compatibility problems.

---

# ▶️ Running the Project

Start Jupyter:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

Open:

```text
Experiement.ipynb
```

Run the notebook sequentially.

The notebook is organised into the following stages:

```text
Step 0  → Observability
Step 1  → Document Intelligence extraction
Step 2  → Two-stage chunking
Step 3  → Azure AI Search
Step 4  → Semantic reranking
Step 5  → LangGraph orchestration
Step 6  → RAGAS evaluation
```

---

# 🧪 Example Queries

### Clear question

```text
How many days of annual leave does a full-time UK employee get?
```

Expected behaviour:

```text
Retrieve UK evidence
        ↓
Grade evidence
        ↓
Generate answer
        ↓
Attach citation
        ↓
Verify
```

---

### Ambiguous question

```text
How much notice do I need to give for booking leave?
```

Expected behaviour:

```text
The system does NOT guess the jurisdiction.

Are you employed in the UK or Ireland?
```

The notebook demonstrates this clarification route.

---

### Unsupported question

```text
What is the statutory sick pay rate per week in the UK?
```

Expected behaviour:

```text
Retrieved evidence is insufficient
            ↓
Do not hallucinate
            ↓
Return grounded refusal
            ↓
Explain missing information
```

The notebook demonstrates this behaviour as well.

---

# 📈 Why Hybrid Search?

Pure vector search is useful for semantic similarity but can miss important exact terminology.

Pure keyword/BM25 search can struggle when the user's wording differs significantly from the policy language.

Hybrid search combines both:

```text
                 Query
                   │
          ┌────────┴────────┐
          ▼                 ▼
       BM25             Vector Search
          │                 │
          └────────┬────────┘
                   ▼
             Hybrid Ranking
                   │
                   ▼
          Semantic Reranking
                   │
                   ▼
              Top Results
```

This is particularly useful for policy documents where both **meaning** and **exact terms** matter.

---

# 🧩 Why LangGraph?

A simple RAG chain could be represented as:

```text
Question → Retrieve → Generate
```

But this project requires conditional behaviour:

```text
Question
   │
   ▼
Understand
   │
   ├── unclear → Clarify
   │
   └── clear
        │
        ▼
     Retrieve
        │
        ▼
       Grade
        │
        ├── insufficient → Refuse
        │
        └── sufficient
              │
              ▼
           Generate
              │
              ▼
            Verify
```

LangGraph provides an explicit stateful workflow for these branches.

The state contains fields such as:

```python
class RAGState(TypedDict, total=False):
    question: str
    search_query: str
    jurisdiction: str | None
    sources: list[dict]
    answer: str
    citations: list[dict]
    route: str
    verified: bool
```

This makes the application's control flow explicit and easier to reason about.

---

# 🔍 Design Principles

## 1. Don't guess missing context

If jurisdiction matters, ask.

```text
UK vs Ireland
```

is not something the model should silently infer.

---

## 2. Retrieve before generating

The LLM should receive policy evidence rather than relying solely on its pretrained knowledge.

---

## 3. Grade retrieved evidence

Retrieval success does not necessarily mean that the retrieved documents contain enough information.

Therefore:

```text
Retrieve ≠ Sufficient evidence
```

The grading step explicitly checks this.

---

## 4. Refuse when evidence is insufficient

A trustworthy RAG application should be comfortable saying:

```text
I can't answer this reliably from the policy library.
```

rather than generating unsupported information.

---

## 5. Preserve source metadata

Every chunk retains:

```text
source_file
page
section
jurisdiction
effective_date
```

This enables filtering, citation, auditing, and debugging.

---

## 6. Evaluate the RAG system

The project includes a small gold dataset and RAGAS evaluation rather than relying only on manual testing.

---

# ⚠️ Current Scope and Limitations

This repository is currently implemented as a **Jupyter notebook-based reference implementation** rather than a packaged production service.

Current limitations include:

* No REST API layer
* No web frontend
* No automated CI/CD pipeline
* Small HR policy corpus
* Document metadata is maintained in a static registry
* Evaluation dataset is relatively small
* Secrets are supplied through environment variables
* Retrieval and orchestration are contained primarily within the notebook

These limitations make the project intentionally focused on demonstrating the **RAG architecture, retrieval strategy, agentic workflow, grounding, evaluation, and observability**.

---

# 🔮 Potential Production Extensions

The architecture can be extended with:

### API layer

Expose the LangGraph application through:

```text
FastAPI
```

### User interface

Add:

```text
Streamlit
React
Gradio
```

### Authentication

Integrate:

```text
Microsoft Entra ID
```

### Secret management

Move credentials to:

```text
Azure Key Vault
```

### Managed identity

Replace API keys with Azure RBAC / managed identity where supported.

### Document ingestion pipeline

Automate:

```text
Document upload
      ↓
Document Intelligence
      ↓
Chunking
      ↓
Embedding
      ↓
Azure AI Search
```

### Continuous evaluation

Run RAGAS evaluation automatically as part of CI/CD.

### Regression testing

Maintain a golden dataset covering:

* Retrieval failures
* Wrong jurisdiction
* Missing evidence
* Hallucination cases
* Citation errors
* Boundary cases

---

# 🏗️ Production-Oriented RAG Architecture

A production evolution of this project could look like:

```text
                         ┌─────────────────┐
                         │   User / UI     │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    FastAPI      │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    LangGraph    │
                         │   Orchestrator  │
                         └────────┬────────┘
                                  │
              ┌───────────────────┼──────────────────┐
              │                   │                  │
              ▼                   ▼                  ▼
         Understand            Retrieve          Verify
              │                   │                  │
              │          ┌────────┴────────┐         │
              │          ▼                 ▼         │
              │        BM25             Vector      │
              │          │                 │         │
              │          └────────┬────────┘         │
              │                   ▼                  │
              │            Semantic Ranker           │
              │                   │                  │
              │                   ▼                  │
              │            Azure AI Search            │
              │                                      │
              └──────────────────┬───────────────────┘
                                 │
                                 ▼
                           Grounded Answer
                                 │
                         ┌───────┴────────┐
                         ▼                ▼
                      Citations       RAGAS
                                         │
                                         ▼
                                      Logfire
```

---

# 📚 Evaluation Dataset

The current evaluation set includes questions covering both UK and Ireland policies.

Examples include:

```text
How many days of annual leave does a full-time UK employee get?

How many public holidays are there in Ireland and are they part of annual leave?

What happens if a UK employee is sick during booked annual leave?

How long does a final written warning stay active in the UK disciplinary procedure?
```

The expected references are defined alongside the evaluation dataset in the notebook.

---

# 🔐 Security Notes

This repository requires credentials for external Azure/OpenAI services.

Do **not** commit:

```text
.env
API keys
Azure Search admin keys
OpenAI keys
Logfire tokens
LangSmith API keys
```

If a credential is accidentally committed, revoke and regenerate it immediately.

---

# 📖 Learning Outcomes

This project demonstrates practical implementation of:

* Retrieval-Augmented Generation
* Enterprise document processing
* Azure Document Intelligence
* Structure-aware chunking
* Semantic chunking
* Text embeddings
* Vector search
* BM25 search
* Hybrid retrieval
* HNSW vector indexing
* Semantic reranking
* Metadata filtering
* LangChain
* LangGraph
* Structured LLM output
* Conditional agentic workflows
* Grounded generation
* Citation generation
* Evidence grading
* Hallucination mitigation
* RAGAS evaluation
* LLM observability
* Production-oriented RAG design

---

# ⭐ Key Takeaway

The central idea behind this project is:

> **A reliable RAG system is not simply "retrieve documents and ask an LLM to answer."**

Instead, it should control the entire decision process:

```text
Understand the question
        ↓
Resolve ambiguity
        ↓
Retrieve relevant evidence
        ↓
Filter by metadata
        ↓
Rerank evidence
        ↓
Assess evidence sufficiency
        ↓
Generate a grounded answer
        ↓
Attach citations
        ↓
Verify the answer
        ↓
Evaluate the system
```

This architecture provides a strong foundation for building **auditable, grounded and enterprise-ready RAG applications**.

---

## 📌 Repository

**GitHub:**
https://github.com/nmanuvenugopal/hr-policy-rag-azure-langgraph

---

## 👤 Author

**Manu Venugopalan**

AI / ML Engineer focused on production AI, NLP, LLM, RAG and agentic AI systems.

---

## 📄 License

This project is provided for educational and demonstration purposes.
