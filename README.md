# Entity Deduplication — Hybrid Methods Benchmark

This repository explores **hybrid entity deduplication** methods by combining **text-based embeddings** (sentence transformers) with **graph-based embeddings** (Node2Vec, NetMF, TransE, DistMult, etc.).  
The codebase is designed to be **modular**, with reusable components for data handling, embedding, similarity computation, and evaluation.

At present, the focus is on the **duplicate detection step** — identifying whether two records refer to the same real-world entity.  
However, the framework is built to be **extensible**, and can later be expanded toward **duplicate resolution** with **human-in-the-loop (HITL)** approaches. This would allow detected duplicates to be merged or curated interactively, enabling practical integration into real-world workflows.

---

## Key Features

- **Modular architecture**  
  All core functionality lives in `modular_methods/`, making it easy to plug in new embedding models, fusion strategies, or evaluation procedures.

- **Hybrid deduplication**  
  Combines sentence embeddings with graph embeddings.

- **Focused but extensible**  
  Current emphasis: *duplicate detection*.  
  Future directions: *duplicate resolution with HITL support*.

---

## Modules Overview

The repository is organized around modular components that can be flexibly combined in different pipelines:

- **`dedup_pipeline.py`**  
  The main orchestration logic for running deduplication experiments.  
  - Extracts entity texts from graphs.  
  - Computes embeddings (text, graph, or hybrid).  
  - Matches entities with cosine similarity and thresholds.  
  - Applies optional literal-based filtering.  
  - Outputs candidate duplicate pairs.

- **`embedding_utils.py`**  
  Provides methods for generating graph and hybrid embeddings.  
  - **Node2Vec** and **NetMF** (graph structure embeddings).  
  - **TransE** and **DistMult** (knowledge graph embeddings via PyKEEN).  
  - Utilities to combine text and graph embeddings into hybrid vectors.

- **`graphToText_utils.py`**  
  Bridges graph data and textual representations.  
  - Extracts labels and literals from RDF graphs.  
  - Groups entities by type.  
  - Supports traversal of graphs to obtain attribute-value pairs for embedding.

- **`similarity_utils.py`**  
  Functions for computing similarity and post-processing matches.  
  - Cosine similarity between embeddings.  
  - Entity matching with thresholds and top-k filtering.  
  - Literal-based comparison with Levenshtein distance and acronym matching.  
  - Adaptive thresholds based on number of common attributes.

- **`output_utils.py`**  
  Formats and structures deduplication results.  
  - Enriches matches with entity details and predicates.  
  - Classifies duplicates into categories (exact, near-exact, similar, conflict).  
  - Prepares JSON-like results for downstream analysis or visualization.


## Models Implemented within the modular methods

This repository currently supports five core embedding models for entity representation and deduplication:

- **Sentence Embeddings**  
  Encodes textual attributes of entities (e.g., names, addresses) using a pretrained Sentence Transformer model.  
  Provides strong performance on string-based duplicates, especially in multilingual contexts.

- **Node2Vec**  
  Learns structural embeddings from the graph by simulating biased random walks.  
  Captures local and global graph neighborhoods, useful for entities linked via organizational structures.

- **NetMF**  
  Matrix factorization method that approximates DeepWalk.  
  Generates embeddings based on high-order proximity, capturing richer relational patterns than Node2Vec alone.

- **TransE**  
  A translational knowledge graph embedding model.  
  Represents relations as vector translations (`h + r ≈ t`), effective for simple relation patterns.

- **DistMult**  
  A bilinear knowledge graph embedding model.  
  Captures symmetric relational patterns and is often strong in link prediction tasks.

Sentence Embeddings are specifically used individually the other in general in **hybrid form**, where text and graph embeddings are fused via α-weighted sum.

### Baseline Model: Dedupe

**Dedupe** is a well-established Python library for record linkage and entity resolution based on **active learning**.  
It builds a classifier that learns how to identify duplicate records by interactively labeling examples.  

- **Strengths**:  
  - Strong baseline performance on tabular/structured data.  
  - Actively learns which fields (e.g., name, address, phone) are most important.  
  - Provides explainability and human-in-the-loop integration.  

- **Limitations**:  
  - Less suited for graph-structured data where relational context matters.  
  - Requires manual labeling or ground truth pairs to start training.  

We include Dedupe as a **baseline comparator** against embedding-based and hybrid methods, ensuring that modern approaches are evaluated against a widely used, production-grade deduplication toolkit. Dedupe was implemented through jupiter notebooks

Each model has a Run_XXXXX File that uses the specific modules needed to make them work.

## Outputs and Evaluation

### Outputs
The modular pipeline produces structured results after running deduplication.  
Each match is enriched with:
- **Entity details**: source graph, subject URI, and associated predicates/attributes.  
- **Similarity scores**: embedding-based cosine similarity and (optionally) literal-level similarity.  
- **Duplication type**: classified as a type of duplicate based on the similarity scores.
  

Results are exported as JSON-like structures, making them easy to inspect or feed into downstream workflows. These results are saved in the matches folder 

---

### Evaluation
To measure effectiveness, we evaluate the models against a **ground-truth golden standard** of known duplicates, that can be found in the data folder.  
Key aspects include:

- **Precision, Recall, F1**  
  Core metrics to quantify accuracy of duplicate detection.  
- **Per-entity analysis**  
  Evaluation is performed across entity types (e.g., HealthcareOrganization, ServiceDepartment, Personnel).  
- **Noise-specific evaluation**  
  Models are stress-tested under controlled data variations:  
  - *Completeness noise* (missing attributes)  
  - *Relational noise* (altered graph links)  
  - *Syntactic variations* (typos, abbreviations, translations)  

- **Runtime and scalability**  
  Time taken to generate embeddings and run similarity checks is logged, allowing efficiency comparisons across models.

## Example Run

To illustrate the workflow, here is a run using the **Hybrid + TransE** setup.

### Input
- **Graphs found in data**
  - `data/healthcare_graph_Main.ttl` – primary knowledge graph  
  - `data/master_data.ttl` – reference/master data  
  - `data/healthcare_graph_struct_low.ttl` – noisy/perturbed graph (structural noise, low level)  

- **Model settings**
  - Sentence encoder: `paraphrase-multilingual-MiniLM-L12-v2`  
  - Graph embedding model: `TransE` (via PyKEEN)  
  - Hybrid weight: `alpha = 0.5`  
  - Embedding dimension: `384`  
  - Matching: cosine similarity, threshold = `0.5`, top_k = `5`  
  - Post-filtering: Levenshtein literal check (predicate-aware, acronym boost)  

### Process (under the hood)

1. **Compute graph embeddings** with TransE.  
2. **Compute sentence embeddings** for entity texts.  
3. **Combine** them into hybrid vectors (α-weighted sum).  
4. **Calculate cosine similarity** between entities.  
5. **Apply literal-level filtering** (Levenshtein + acronyms, adaptive thresholds).  
6. **Format matches** with entity details and assign duplication labels.  

---

### Output

- **Matches (JSON):**  
  `matches/matches_struct_low/HybridTransE_alpha_0.5.json`  
  Each entry includes:
  - The two matched entities (URIs + selected predicates)  
  - Similarity scores (embedding and literal)  
  - Duplication type: `true_duplicate`, `near-exact`, `similar`, or `conflict`  

- **Runtime log:**  
  `runtimes.txt` with total execution time.  

## 🔮 Future Extensions

The current module specification provides a strong foundation for entity deduplication, but several areas remain to be implemented or extended:

- **REST API implementation**  
  While the API endpoints (`/execute` and `/hitl`) are defined, the actual service layer still needs to be built and exposed (e.g., via FastAPI or Flask). This includes request validation, error handling, and standardized JSON responses.

- **Dockerization**  
  A containerized deployment (Dockerfile, docker-compose) is required to meet the non-functional requirements and ensure portability across environments.

- **Duplicate classification**  
  Logic for categorizing duplicates into *exact*, *similar*, and *conflict* is implemented, but needs to be refined to properly allow a conflict-resolution mechanism (auto vs. HITL escalation) to work

- **HITL integration**  
The Human-in-the-Loop workflow (logging, dashboard, or user-facing UI for conflict resolution) still needs to be connected to the backend so that flagged conflicts can be reviewed and resolved interactively.

- **SPARQL query generation**  
  Auto-resolution requires generating valid SPARQL `DELETE/INSERT` queries. While example outputs are included in the specification, dynamic query construction is not yet implemented.


- **Input/Output adapters**  
  Functions to fetch graphs from a triple store (via GraphDB API or SPARQL) and to push the resolved graphs back need to be implemented. Currently, only placeholders are provided in the specification.


- **Integrated Logging and monitoring**  
  A structured logging system and basic monitoring/metrics collection (e.g., number of duplicates found, resolution types) would make the module production-ready.

---

These extensions represent the next development milestones required to move from the current design to a fully working, deployable deduplication module.



