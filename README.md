# Entity De-duplication Testing  Specifications

## Sections

- Definitions
- 1.Description/Specification
  - 1.1 Scope
  - 1.2 Requirements
  - 1.3 Assumptions related to implementation
  - 1.4 Method
- 2.Design and architecture including data flows
  - 2.1 Architecture and Data flows
  - 2.2 Input and Output Description

## Definitions

**Master Data**: refers to the core business information that is critical for
maintaining consistency and accuracy across an organization. It includes data
elements such as customer information, product details, supplier records, and
location specifics. Master Data Management (MDM) ensures this foundational data
is consistent, accurate, and accessible throughout the enterprise, facilitating
better decision-making and improving operational efficiency. By centralizing and
standardizing master data, organizations can reduce errors and enhance data
integrity.

**Master Entity**: refers to a specific item or object within the realm of
Master Data that represents a unique individual or concept in the organization’s
database. It is crucial for ensuring each entity (e.g., hospital, personnel) is
uniquely identifiable and consistently represented throughout the enterprise.
Master Entities are essential for maintaining accurate relationships between
different pieces of data.

**Source Knowledge Graph (SKG)**: refers to a structured representation of
knowledge extracted from various sources within an organization. SKGs capture
relationships and attributes of entities as they exist in their original source
systems. These graphs are later converted into a common format and merged into
the Personal Health Knowledge Graph (PHKG).

**Personal Health Knowledge Graph (PHKG)**: is derived by merging all Source
Knowledge Graphs (SKGs) from various data sources. The PHKG provides a
comprehensive view of personal health information, capturing relationships and
attributes related to individual patients or users within healthcare
organizations.

**Deduplication**: involves identifying and removing duplicate entries in the
Personal Health Knowledge Graph (PHKG). This process ensures that each record is
unique and accurate, enhancing data integrity and consistency.

**Text Embeddings**: refer to numerical representations of text data that
capture semantic meaning and context. These embeddings enable text to be
processed numerically, facilitating tasks like similarity searches,
classification, and clustering. Text embeddings are generated using machine
learning models and are crucial for natural language processing applications,
such as information retrieval and sentiment analysis.

**Graph Embedding**: refers to the process of representing nodes and edges of a
graph in a low-dimensional vector space while preserving the structural and
semantic properties of the original graph. This involves converting complex
relationships within a graph into numerical vectors that can be easily analyzed
and processed using machine learning algorithms. Graph embeddings are crucial
for tasks such as node classification, link prediction, community detection, and
anomaly detection. By capturing both local and global structures of the graph,
these embeddings enable more effective analysis and decision-making in various
domains, including social networks, recommendation systems, and healthcare
analytics.

**Cosine Similarity**: measures the similarity between two non-zero vectors in
an inner product space. It calculates the cosine of the angle between two
vectors, with values ranging from -1 (completely dissimilar) to 1 (identical).
This metric is widely used in tasks like information retrieval and
recommendation systems to determine how closely related two pieces of data are.

**HITL (Human in the Loop)**: involves integrating human expertise and judgment
into automated systems. In HITL, critical decisions or tasks are reviewed and
validated by human experts, ensuring that automation is balanced with human
oversight. This approach is particularly important in complex systems where
errors could have significant consequences, ensuring data accuracy and alignment
with organizational goals.

## 1. Description/Specification

### 1.1 Scope

This document outlines the requirements and methodology for the entity
deduplication module, which is designed to ensure each Master Entity (such as an
Organization, Person, Health Care Provider, Address, etc.) is uniquely
identifiable and consistently represented throughout the enterprise. The primary
objectives of this module are as follows:

1. **Identify and Resolve Duplicates**: The module will identify duplicate
   records across various Master Entities and resolve them by linking duplicates
   to appropriate Master Entities. This ensures that each entity is accurately
   and consistently represented.

2. **Link to Master Data**: Once duplicates are identified and resolved, the
   module will link these entities back to their corresponding Master Data
   entries. This integration enhances data integrity and supports accurate
   business processes across different systems.

By achieving these objectives, the entity deduplication module aims to maintain
high standards of data accuracy and consistency, reducing errors and improving
operational efficiency. The module is designed to be an ongoing process,
ensuring that new data continues to be integrated and duplicates remain resolved
over time. This integration provides a robust foundation for various enterprise
processes and applications.

This approach ensures that all Master Entities are accurately managed and
consistently represented across the organization, supporting better
decision-making and enhancing overall operational efficiency.

### 1.2 Requirements

#### Non-functional requirements

- The module can be implemented in **any** specified programming language (e.g.,
  Python, Java).
- The module must be delivered as a **Docker** container.
- The module should expose a **REST API** for input and output, adhering to
  specific endpoints and data formats.
- The module should process a X number of entities in Y time.
- The module should be compatible with various hospital master data structures
  without requiring additional retraining or retuning. Configuration is allowed
  for customization.
- Module should be able work with empty/non-complete master data.

#### Functional requirements

- Module should support the following languages:

  - Dutch
  - German
  - Estonian
  - English

- Module should find duplicates between two RDF graphs and return similarity
  scores using a predefined method.

- Module should identify conflicts among the found duplicates based on
  predefined criteria (e.g., semantic differences, structural anomalies).

- The module should determine if a conflict can be resolved automatically.

  - If it can, it should generate SPARQL queries to resolve the conflict.
  - Otherwise, it should create a Human In The Loop (HITL) entry to escalate the
    issue to the user.

- Module should identify and add entities not present in the Master Data but
  found in the new SKG, ensuring they meet specified quality standards.

- Module is expected to adhere to defined input/output formats in this document.

- In any case, module should not completely delete statements from the graphs.
  - Module can mark statements deleted or move deleted statements to another
    graph.

### 1.3 Assumptions related to implementation

It is assumed that this module will be used for processing and managing
**NON-MEDICAL** entities. These entities include Organizations, Individuals
(Persons), Addresses, and other similar data types.

In case of its use in **MEDICAL** entities, stringent requirements for accuracy
and reliability are needed.

## 2.Design and architecture including data flows

### 2.1 Architecture and Data flows

### 2.2 Input and Output Description

This module will only expose two endpoints:

#### Endpoint `/execute`

This endpoint will be used to execute the entity deduplication process.

##### Input for `/execute`

The input will consist of following fields:

- `phkg_graph_name`: Name of the Personal Health Knowledge Graph (PHKG) to be
  processed.
- `skg_graph_name`: Name of the Source Knowledge Graph (SKG) to be processed.
- `master_data_graph_name`: Name of the Master Data Graph to be used for
  resolving duplicates.

Application should use SPARQL queries or GraphDB API to fetch the graphs from
the triple store.

Below is an example of the input json:

```json
{
  "phkg_graph_name": "http://example.com/phkg",
  "skg_graph_name": "http://example.com/patient/123/data/1",
  "master_data_graph_name": "http://example.com/master_data"
}
```

##### Output for `/execute`

The output will consist of an array of objects, each object representing a
duplicate pair with the following fields:

- `entities`: Array of objects, each representing an entity in the duplicate
  pair. Can consist of two or more entities from same or different graphs.
  - `entity1`: Object representing the first entity.
    - `from`: Graph name from which the entity is coming.
    - `subject`: URI of the entity.
    - `predicate`: Predicate of the entity.
    - `object`: Object of the entity.
  - `entity2`: Object representing the second entity.
    - `from`: Graph name from which the entity is coming.
    - `subject`: URI of the entity.
    - `predicate`: Predicate of the entity.
    - `object`: Object of the entity.
- `similarity_score`: Similarity score between the two entities.
- `duplication_type`: Type of duplicate pair. Can be one of the following:
  - `exact`: Exact duplicate. There might be some differences in the entities
    but they are considered same. E.g., "John Doe" and "John Doe ".
  - `similar`: Similar but not exact duplicate. These can be different
    representations of same thing. E.g., "John Doe" and "Doe, John".
  - `conflict`: Conflict between the two entities.
- `duplication_resolution`: Resolution of the conflict. Can be one of the
  following:
  - `auto`: Conflict will be resolved automatically using provided SPARQL
    queries.
  - `hitl`: Conflict needs human intervention. A HITL entry will be created.
- `auto_resolution_data`: SPARQL queries to resolve the conflict. This field
  will be present only if `duplication_resolution` is `auto`.
  - `apply`: SPARQL query to apply to the graph.
  - `undo`: SPARQL query to undo the changes if needed.
- `hitl_resolution_data`: HITL entry data. This field will be present only if
  `duplication_resolution` is `hitl`. This field should contain all the
  necessary data to create the HITL entry and apply the resolution. The given
  structure is not final and can be adjusted based on the requirements.
  - `subject_label`: Label of the subject entity.
  - `predicate_label`: Label of the predicate.
  - `duplicate_objects`: Array of objects, each representing the object of the
    entity.
    - `from`: Graph name from which the entity is coming.
    - `subject`: URI of the entity.
    - `predicate`: Predicate of the entity.
    - `object`: Object of the entity.

Below you can see few example outputs:

```json
[
  {
    "entities": [
      {
        "entity1": {
          "from": "http://example.com/patient/123/data/1",
          "subject": "http://example.com/patient/123",
          "predicate": "http://example.com/hasName",
          "object": "John Doe "
        },
        "entity2": {
          "from": "http://example.com/master_data",
          "subject": "http://example.com/patient/124",
          "predicate": "http://example.com/hasName",
          "object": "John Doe"
        }
      }
    ],
    "similarity_score": 0.9,
    "duplication_type": "exact",
    "duplication_resolution": "auto",
    "auto_resolution_data": {
    "apply": """
        DELETE {
         GRAPH <http://example.com/patient/124/data/1> {
           <http://example.com/patient/124> <http://example.com/hasName> 'John Doe' }
         }
        INSERT {
         GRAPH <http://example.com/patient/124/data/1_deleted> {
           <http://example.com/patient/124> <http://example.com/hasName> 'John Doe'
         }
         GRAPH <http://example.com/patient/123/data/1> {
           <http://example.com/patient/123> <http://example.com/hasName> 'John Doe'
         }
        }",
    "undo": """
        DELETE {
         GRAPH <http://example.com/patient/124/data/1_deleted> {
           <http://example.com/patient/124> <http://example.com/hasName> 'John Doe'
         }
         GRAPH <http://example.com/patient/123/data/1> {
           <http://example.com/patient/123> <http://example.com/hasName> 'John Doe'
         }
        }
        INSERT {
         GRAPH <http://example.com/patient/124/data/1> {
           <http://example.com/patient/124> <http://example.com/hasName> 'John Doe'
         }
        }"""
    },
    "hitl_resolution_data": null
  }
]
```

```json
[
  {
    "entities": [
      {
        "entity1": {
          "from": "http://example.com/patient/123/data/1",
          "subject": "http://example.com/patient/123",
          "predicate": "http://example.com/hasName",
          "object": "John Doe"
        },
        "entity2": {
          "from": "http://example.com/master_data",
          "subject": "http://example.com/patient/124",
          "predicate": "http://example.com/hasName",
          "object": "J. Doe"
        }
      }
    ],
    "similarity_score": 0.7,
    "duplication_type": "similar",
    "duplication_resolution": "hitl",
    "auto_resolution_data": null,
    "hitl_resolution_data": {
      "subject_label": "Patient",
      "predicate_label": "name",
      "duplicate_objects": [
        {
          "from": "http://example.com/patient/123/data/1",
          "subject": "http://example.com/patient/123",
          "predicate": "http://example.com/hasName",
          "object": "John Doe"
        },
        {
          "from": "http://example.com/master_data",
          "subject": "http://example.com/patient/124",
          "predicate": "http://example.com/hasName",
          "object": "J. Doe"
        }
      ]
    }
  }
]
```

#### Endpoint `/hitl`

This endpoint is responsible for creating SPARQL queries to the given HITL
answers.

##### Input for `/hitl`

The input is an array of objects, each object representing one entity that needs
to be deleted or added to the given target graph.

Note: This endpoint might change based on the requirements.

- `target_graph`: Name of the target graph to apply the changes.
- `subject`: URI of the entity.
- `predicate`: Predicate of the entity.
- `object`: Object of the entity.
- `action`: Action to be performed. Can be one of the following:
  - `delete`: Delete the entity from the target graph.
  - `add`: Add the entity to the target graph.

Below is an example of the input json:

```json
[
  {
    "target_graph": "http://example.com/master_data",
    "subject": "http://example.com/patient/124",
    "predicate": "http://example.com/hasName",
    "object": "Doe, John",
    "action": "add"
  }
]
```

```json
[
  {
    "target_graph": "http://example.com/patient/123/data/1",
    "subject": "http://example.com/patient/123",
    "predicate": "http://example.com/hasName",
    "object": "John Due",
    "action": "delete"
  },
  {
    "target_graph": "http://example.com/patient/123/data/1",
    "subject": "http://example.com/patient/123",
    "predicate": "http://example.com/hasName",
    "object": "John Doe",
    "action": "add"
  }
]
```

##### Output for `/hitl`

Array of SPARQL queries to apply and undo the changes.

Below is an example of the output:

```json
[
  {
    "apply": """
      DELETE {
        GRAPH <http://example.com/patient/123/data/1> {
          <http://example.com/patient/123> <http://example.com/hasName> 'John Due'
        }
      }
      INSERT {
        GRAPH <http://example.com/patient/123/data/1/deleted> {
          <http://example.com/patient/123> <http://example.com/hasName> 'John Due'
        }
        GRAPH <http://example.com/patient/123/data/1> {
          <http://example.com/patient/123> <http://example.com/hasName> 'John Doe'
        }
      }""",
    "undo": """
      DELETE {
        GRAPH <http://example.com/patient/123/data/1/deleted> {
          <http://example.com/patient/123> <http://example.com/hasName> 'John Due'
        }
        GRAPH <http://example.com/patient/123/data/1> {
          <http://example.com/patient/123> <http://example.com/hasName> 'John Doe'
        }
      }
      INSERT {
        GRAPH <http://example.com/patient/123/data/1> {
          <http://example.com/patient/123> <http://example.com/hasName> 'John Due'
        }
      }"""
  }
]
```