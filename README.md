# Embedding Tools

This project provides a set of tools for generating and managing embeddings using different models, such as BioGPT and BGE (BAAI General Embeddings). It allows you to perform various tasks like generating embeddings for a repository of strings, clustering embeddings, assigning labels to new embeddings, and matching queries against a repository.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/embedding-tools.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

The main classes provided in this project are:

- `embed_task_with_repo`: A class for embedding tasks with a repository of strings.
- `short_embed_tast`: A class for short embedding tasks without a repository.

### Embedding Task with Repository

1. Initialize the `embed_task_with_repo` class with the desired model and repository:
   ```python
   from embed_task import embed_task_with_repo

   repo_strings = [...]  # Your repository of strings
   embedder = embed_task_with_repo('bge-en', repo_strings)
   ```

2. Generate embeddings for the repository:
   ```python
   embedder.generate_embeddings_for_repo()
   ```

3. Save and load embeddings:
   ```python
   embedder.save_embeddings("embeddings.npy")
   loaded_embedder = embed_task_with_repo('bge-en', repo_strings)
   loaded_embedder.load_embeddings("embeddings.npy")
   ```

4. Perform clustering on the embeddings:
   ```python
   embedder.cluster(method='knn', n_clusters=50, random_state=123, colnameoflabel='test1')
   ```

5. Assign cluster labels to new embeddings:
   ```python
   assigned_labels = embedder.assign_labels_to_new_embeddings(new_strings, 'test1', method='knn')
   ```

6. Match queries against the repository:
   ```python
   query = "..."
   matched_items = embedder.match(query, topk=2)
   ```

### Short Embedding Task

1. Initialize the `short_embed_tast` class with the desired model:
   ```python
   from embed_task import short_embed_tast

   embedder = short_embed_tast('bge-en')
   ```

2. Get similarity scores for pairs of strings:
   ```python
   pairlist = [('string1', 'string2'), ...]
   similarity_scores = embedder.get_similarity_forpair(pairlist)
   ```

## Examples

Detailed examples of how to use the embedding tools can be found in the `Example.ipynb` and `Example1.ipynb` Jupyter Notebook.

## Files

- `BGEEmbedding.py`: Contains the `BGEEmbedding` class for generating embeddings using BGE models.
- `BioGptEmbedding.py`: Contains the `BioGptEmbedding` class for generating embeddings using the BioGPT model.
- `embed_task.py`: Contains the `embed_task_with_repo` and `short_embed_tast` classes for embedding tasks.
- `Example.ipynb`: Jupyter Notebook with examples of how to use the embedding tools.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
