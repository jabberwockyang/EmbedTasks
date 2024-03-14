from __future__ import annotations
from BGEEmbedding import BGEEmbedding
from BioGptEmbedding import BioGptEmbedding

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


class embed_task_with_repo():
    '''
    A class for embedding task


    functions:
    - __init__: initialize the class with metadata and embeddings

    - generate_embeddings_for_repo: generate embeddings for the metadata
    - save_embeddings: save the embeddings to a np file
    - load_embeddings: load the embeddings from a np file
    - get_similarity_distribution: get the distribution of similarity within the repo for downstream task

    - cluster: cluster the repo based on certain similarity threshold
    - remove_overlapping: map the similar items with the representative item in the repo

    - topmatch: find a list of the most similar items in the repo for a given query with a threshold or topk
    - thematch: find the most similar item in the repo for a given query  
    - mapping: map the most similar items in the repo for a list of queries


    '''
    def __init__(self, modelname:str, repo : list| pd.DataFrame = None ,
                 colname: str = None, 
                 embeddings=None, 
                 local_files_only:bool = True):
        '''
        Parameters:

        - modelname (required): A string specifying the name of the model to be used. It can be one of the following: 'biogpt', 'bge-zh', or 'bge-en'.

        - repo (optional): A list of strings or a pandas DataFrame containing the metadata for the repository. If not provided, it defaults to None.

        - colname (optional): A string specifying the name of the column in the DataFrame that contains the strings. It is required if repo is a DataFrame. If not provided, it defaults to None.

        - embeddings (optional): A numpy array containing pre-computed embedding vectors for the repository. If not provided, it defaults to None.

        - local_files_only (optional): A boolean indicating whether to use only local files. If not provided, it defaults to True.

        Logic:
        - for the input repo:
            1. If repo is None, it means no metadata is provided. In this case:

                - self.metadata is set to None.

                - A message is printed indicating that no metadata is found and the instance is initialized without a repository.

            2. If repo is a pandas DataFrame, it means the metadata is provided as a DataFrame. In this case:
                - self.metadata is set to the provided DataFrame.

                - self.__stringcol__ is set to the value of colname, indicating the column name containing the strings.
                
                - If colname is None or not found in the DataFrame columns, a ValueError is raised.
            
                - self.repo is set to the list of strings extracted from the specified column of the DataFrame.

            3. If repo is a list, it means the metadata is provided as a list of strings. In this case:
                
                - The list is converted to a pandas DataFrame with a single column named 'string', and assigned to self.metadata.
                
                - self.__stringcol__ is set to 'string'.
                
                - self.repo is set to the list of strings extracted from the 'string' column of the DataFrame.
                
                - A message is printed indicating that a list of strings is accepted as metadata and the column name is set to 'string'.

            4. If repo is neither None, a DataFrame, nor a list, a ValueError is raised, indicating that the input metadata is not supported.
        
            5. self.dim_of_repo is set to the length of self.repo if it is not None, otherwise it is set to None.

        - for the input embeddings:
            1. self.embeddings is set to the provided embeddings parameter.

            2. self.dim_of_repo_embeddings is set to the shape of self.embeddings if it is not None, otherwise it is set to None.

            3. If embeddings is None, a message is printed indicating that no embeddings are found and the generate_embeddings_for_repo method should be run first.
        
            4. If the first dimension of self.dim_of_repo and self.dim_of_repo_embeddings do not match, a ValueError is raised, indicating that the length of metadata and embeddings do not match.
        
        - for the input modelname:

            - Based on the value of modelname, the corresponding model object is created and assigned to self.model, and self.modelname is set accordingly.
        '''

        if repo is None:
            self.metadata = None
            print('No metadata found, initialized without repo')
        elif isinstance(repo, pd.DataFrame):
            self.metadata = repo 
            self.__stringcol__ = colname
            if colname is None:
                raise ValueError('accepting dataframe as metadata, column name not provided, please check the input')
            elif colname not in self.metadata.columns:
                raise ValueError('accepting dataframe as metadata, column name not found in metadata, please check the input')
            else:
                self.repo = self.metadata[colname].tolist()
        elif isinstance(repo, list):
            self.metadata= pd.DataFrame(repo, columns=['string'])
            self.__stringcol__ = 'string'
            self.repo = self.metadata['string'].tolist()

            print('accepting list of strings as metadata, column name is set to `string`')
        else:
            raise ValueError('Input metadata is not supported, please check the input')
        
        self.dim_of_repo = len(self.repo) if self.repo is not None else None

        
          
        self.embeddings = embeddings
        self.dim_of_repo_embeddings = self.embeddings.shape if self.embeddings is not None else None
        if embeddings is None:
            print('No embeddings found, please run `generate_embeddings_for_repo` first')
          
        elif self.dim_of_repo[0] != self.dim_of_repo_embeddings[0]:
            raise ValueError('Length of metadata and embeddings do not match, please check the input')
        
        if modelname == 'biogpt':
            self.model = BioGptEmbedding(local_files_only = local_files_only)
            self.modelname = 'biogpt'
        elif modelname == 'bge-zh':
            self.model = BGEEmbedding("zh",local_files_only = local_files_only)
            self.modelname = 'bge-zh'
        elif modelname == 'bge-en':
            self.model = BGEEmbedding("en",local_files_only = local_files_only)
            self.modelname = 'bge-en'
        else:
            raise ValueError('Model not found, please check the input')




    def generate_embeddings_for_repo(self,
                                     batch_size: int = 200):
        '''
        input:
        - colname: string, column name of the metadata
        - batch_size: int, batch size for generating embeddings
        output:
        - numpy array of embeddings
        '''
       
        self.embeddings = self.model.generate_embeddings(self.repo, batch_size = batch_size)
        print('Embeddings generated, shape:',self.embeddings.shape, "access with .embeddings")
    
    def save_embeddings(self, filename: str):
        '''
        input:
        - filename: string, filename for saving the embeddings
        '''
        np.save(filename, self.embeddings)
    
    def load_embeddings(self, filename: str):
        '''
        input:
        - filename: string, filename for loading the embeddings
        '''
        self.embeddings = np.load(filename)
        self.dim_of_repo_embeddings = self.embeddings.shape
        
    
    def __cosine_similarity__(self,embeddings1, embeddings2):
        '''
        input:
        - embeddings1: M × 1024 matrix（M non-standard diagnoses, 1024 embedding size）
        - embeddings2: N × 1024 matrix（N standard codes, 1024 embedding size）
        output:
        - similarity_matrix: M × N matrix（M non-standard diagnoses, N standard codes）
        '''
        embeddings1_normalized = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2_normalized = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        similarity_matrix = np.dot(embeddings1_normalized, embeddings2_normalized.T)
        return similarity_matrix

    def get_similarity_distribution(self,plot: bool = True):   
        '''
        identify threshold value for downstream task
        input:
        - plot: boolean, if True, plot the distribution of similarity scores
        output:
        - histogram of similarity scores if plot is True
        - similarity_matrix: numpy array of similarity scores within the repo saved to self.similarity_distribution
        '''
        
        if self.embeddings is None:
            raise ValueError('No embeddings of repo found, please run generate_embeddings first')

        else:
            pass
        
        similarity_matrix = self.__cosine_similarity__(self.embeddings, self.embeddings)
        np.fill_diagonal(similarity_matrix, 0)

        self.similarity_distribution = similarity_matrix
        if plot:
            plt.hist(self.similarity_distribution.flatten(), bins=100)
            plt.show()


    def cluster(self, method:str = 'knn', n_clusters: int = 10, random_state: str = 123,
                eps: float = 0.5, min_samples: int = 5, colnameoflabel: str = None):
        '''
        cluster the repo based on certain similarity threshold

        input:

        - method: string, 'knn' or 'dbscan'
        - n_clusters: int, number of clusters for kmeans
        - random_state: int, random state for kmeans
        - eps: float, maximum distance between two samples for one to be considered as in the neighborhood of the other
        - min_samples: int, number of samples in a neighborhood for a point to be considered as a core point

        output:

        - cluster labels saved to metadata
        '''
        if self.embeddings is None:
            raise ValueError('No embeddings of repo found, please run generate_embeddings first')


        if method == 'knn':
            self.__kmeans_cluster__ = (n_clusters, random_state)
            kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=random_state)
            kmeans.fit(self.embeddings)
            labels = kmeans.labels_
            if colnameoflabel is None:
                colnameoflabel = f'cluster_knn{n_clusters}_rs{random_state}'
            self.metadata[colnameoflabel] = labels
        elif method == 'dbscan':
            self.__dbscan_cluster__ = (random_state)
            
            dist_matrix = 1 - self.similarity_distribution

            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            clusters = dbscan.fit_predict(dist_matrix)
            if colnameoflabel is None:
                colnameoflabel = f'cluster_dbscan_eps{eps}_minsamp{min_samples}'
            self.metadata[colnameoflabel] = clusters
        else:
            raise ValueError('Invalid method, please use knn or dbscan')
        
    def assign_labels_to_new_embeddings(self, new_strings: list, label_col: str, method: str = 'knn') -> list[int]:
        """
        Assign cluster labels to new embeddings based on the specified clustering method.

        :param new_strings: The list of new strings to be assigned cluster labels.
        :param label_col: The name of the column in metadata dataframe that contains the cluster labels.
        :param method: The clustering method used. Can be 'knn' or 'dbscan'.
        :return: A list of cluster labels for the new embeddings.
        """

        if label_col not in self.metadata.columns:
            raise ValueError(f"Column {label_col} not found in metadata.")
        
        if method not in ['knn', 'dbscan']:
            raise ValueError(f"Unsupported clustering method: {method}. Must be 'knn' or 'dbscan'.")
        
        new_embeddings = self.model.generate_embeddings(new_strings)

        if method == 'knn':
            # Compute the centroids for each cluster
            unique_labels = self.metadata[label_col].unique()
            centroids = []
            for label in unique_labels:
                cluster_embeddings = self.embeddings[self.metadata[label_col] == label]
                centroid = np.mean(cluster_embeddings, axis=0)
                centroids.append(centroid)
            
            # Compute the similarity between each new embedding and each centroid
            similarity_matrix = self.__cosine_similarity__(new_embeddings, np.array(centroids))
            
            # Assign each new embedding to the cluster with the most similar centroid
            assigned_labels = [unique_labels[np.argmax(row)] for row in similarity_matrix]
        
        elif method == 'dbscan':
            # Compute the distance matrix between new embeddings and all original embeddings
            distance_matrix = 1 - self.__cosine_similarity__(new_embeddings, self.embeddings)
            
            # For each new embedding, find the label of the nearest core sample
            assigned_labels = []
            for i in range(len(new_embeddings)):
                nearest_core_idx = np.argmin(distance_matrix[i])
                nearest_core_label = self.metadata.at[nearest_core_idx, label_col]
                assigned_labels.append(nearest_core_label)
        
        return assigned_labels
    

    def remove_overlapping(self, label_col:str):
        '''
        get the overlapping items in the repo without an extra query
        input:
        - method: string, 'rank' or 'cutoff'
        - threshold: float, threshold value for cutoff method
        - topk: int, number of top results to return
        - return_sim: bool, if True, return similarity scores
        output:
        - dataframe of overlapping items
        '''

        X = self.embeddings
        sentences = self.metadata[self.__stringcol__]
        labels = self.metadata[label_col]
        n_clusters = len(np.unique(labels))
        n_features = X.shape[1]
        cluster_centers = np.zeros((n_clusters, n_features))

        # Calculate the cluster centers
        for i in range(n_clusters):
            cluster_centers[i, :] = X[labels == i].mean(axis=0)

        representative_sentence_map = {}
        
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            distances = np.linalg.norm(cluster_points - cluster_centers[i], axis=1)
            representative_idx = np.argmin(distances)
            # representative_vector = cluster_points[representative_idx]
            representative_sentence = sentences[labels == i][representative_idx]
            representative_sentence_map[i] = representative_sentence
            self.metadata['representative_vector'] = self.metadata[label_col].apply(lambda x: representative_sentence_map[x])
            self.__representative_sentence_map__ = representative_sentence_map
            
    def rerank_topk(self, query, sim_scores, topk):
        if self.modelname not in ['bge-zh', 'bge-en']:
            raise ValueError('Reranking is only supported for BGE models.')
        
        top2k = np.argsort(sim_scores)[-topk*2:][::-1]
        pairs = [[query, self.repo[i]] for i in top2k]
        reranked_scores = self.model.rerank(pairs)
        reranked_indices = np.argsort(reranked_scores)[-topk:][::-1]
        return [self.repo[i] for i in reranked_indices]
    
    def match(self, queries: str | list, threshold: float = None, topk: int = None, reranker: bool = False, return_sims: bool = False):
        """
        Find the most similar items in the repo for given queries.
        
        :param queries: A single query or a list of queries.
        :param threshold: The similarity threshold. If not None, return items with similarity higher than this threshold.
        :param topk: The number of top items to return. If not None, return the topk most similar items.
        :param reranker: Whether to use the reranker. Only applicable to BGE models.
        :param return_sims: Whether to return the similarity scores along with the items.
        :return: If return_sims is False, return a list of most similar items. If queries is a single query, return a list of items. 
                If queries is a list of queries, return a list of lists of items.
                If return_sims is True, return a tuple (items, sims), where items is as described above, and sims is a list of 
                corresponding similarity scores.
        """
        if self.embeddings is None:
            raise ValueError('No embeddings found. Please run generate_embeddings first.')
        
        if not isinstance(queries, (str, list)):
            raise ValueError('Queries must be either a string or a list of strings.')
        
        if isinstance(queries, str):
            queries = [queries]
        
        if reranker and self.modelname not in ['bge-zh', 'bge-en']:
            raise ValueError('Reranking is only supported for BGE models.')
        
        query_embeddings = self.model.generate_embeddings(queries)
        similarity_matrix = self.__cosine_similarity__(query_embeddings, self.embeddings)

        results = []
        sims_list = []
        for sims in similarity_matrix:
            if threshold is not None:
                indices = np.where(sims >= threshold)[0]
            elif topk is not None:
                indices = np.argsort(sims)[-topk:][::-1]
            else:
                raise ValueError('Either threshold or topk must be provided.')
            
            if reranker:
                reranked_indices = self.rerank_topk(queries[len(results)], sims, len(indices))
                results.append([self.repo[i] for i in reranked_indices])
                if return_sims:
                    sims_list.append([sims[i] for i in reranked_indices])  # 使用原始的相似度分数
            else:
                results.append([self.repo[i] for i in indices])
                if return_sims:
                    sims_list.append([sims[i] for i in indices])

        if len(queries) == 1:
            return (results[0], sims_list[0]) if return_sims else results[0]
        else:
            return (results, sims_list) if return_sims else results


            

    def match_results_to_dict(self, match_results: list[list[str]], queries: list[str]) -> dict[str, list[str]]:
        """
        Convert the results from match method to a dictionary format.

        :param match_results: The results from match method, a list of lists of strings.
        :param queries: The original queries, a list of strings.
        :return: A dictionary where keys are queries and values are lists of matched items.
        """
        if len(match_results) != len(queries):
            raise ValueError("The number of match results must be equal to the number of queries.")
        
        return {query: matched_items for query, matched_items in zip(queries, match_results)}

class short_embed_tast():
    '''
    A class for embedding task
    '''
    def __init__(self, modelname:str, local_files_only:bool = True):
        if modelname == 'biogpt':
            self.model = BioGptEmbedding(local_files_only = local_files_only)
            self.modelname = 'biogpt'
        elif modelname == 'bge-zh':
            self.model = BGEEmbedding("zh",local_files_only = local_files_only)
            self.modelname = 'bge-zh'
        elif modelname == 'bge-en':
            self.model = BGEEmbedding("en",local_files_only = local_files_only)
            self.modelname = 'bge-en'
        else:
            raise ValueError('Model not found, please check the input')
    
    def __cosine_similarity_forpair__(self,matrix1, matrix2):
        dot_product = np.dot(matrix1, matrix2.T)
        norm_matrix1 = np.linalg.norm(matrix1, axis=1, keepdims=True)
        norm_matrix2 = np.linalg.norm(matrix2, axis=1, keepdims=True)
        return np.diagonal(dot_product / (np.dot(norm_matrix1, norm_matrix2.T)))

 
    def get_similarity_forpair(self, pairlist: list,batch_size = 200):
        '''
        
        input:
        - pairlist: list of pairs of strings
        - batch_size: int, batch size for generating embeddings
        output:
        - similarity score of each pair of strings 
        '''
        string1 = [pair[0] for pair in pairlist]
        string2 = [pair[1] for pair in pairlist]
        embeddings1 = self.model.generate_embeddings(string1, batch_size=batch_size)
        embeddings2 = self.model.generate_embeddings(string2, batch_size=batch_size)
        similarity_scores = self.__cosine_similarity_forpair__(embeddings1, embeddings2)
        return similarity_scores

      
            
        
        

# Example usage
        
if __name__ == '__main__':
    # Example usage 1: Initializing with a list of strings
    repo_strings = [
            "Coronary artery disease is caused by plaque buildup in the arteries supplying blood to the heart.",
            "Type 2 diabetes mellitus is a chronic condition characterized by high blood sugar levels due to insulin resistance.",
            "Hypertension, or high blood pressure, is a major risk factor for cardiovascular diseases.",
            "Asthma is a respiratory condition that causes inflammation and narrowing of the airways.",
            "Chronic obstructive pulmonary disease (COPD) is a group of lung diseases that cause airflow blockage and breathing problems."
        ]
    embedder = embed_task_with_repo('bge-en', repo_strings)
    embedder.generate_embeddings_for_repo()
    
    # Example usage 2: Saving and loading embeddings
    embedder.save_embeddings("embeddings.npy")
    loaded_embedder = embed_task_with_repo('bge-en', repo_strings)
    loaded_embedder.load_embeddings("embeddings.npy")
    del loaded_embedder

    # Example usage 3: Initializing with a DataFrame
    repo_df = pd.DataFrame({"text": [
        "Alzheimer's disease is a progressive neurological disorder that affects memory and cognitive functions.",
        "Parkinson's disease is a movement disorder characterized by tremors, stiffness, and difficulty with balance and coordination.",
        "Multiple sclerosis is an autoimmune disease that attacks the protective covering of nerve fibers, causing communication problems between the brain and the body.",
        "Epilepsy is a neurological disorder characterized by recurrent and unprovoked seizures.",
        "Migraine is a neurological condition that causes severe headaches, often accompanied by sensitivity to light and sound."
    ]})
    embedder_df = embed_task_with_repo('bge-en', repo_df, colname='text')
    embedder_df.generate_embeddings_for_repo()
    del embedder_df

    # Example usage 4: Getting similarity distribution
    embedder.get_similarity_distribution(plot=True)
    print(embedder.similarity_distribution.shape)
    print(embedder.similarity_distribution)

    # Example usage 5: Clustering
    embedder.cluster(method='knn', n_clusters=3, random_state=123, colnameoflabel='test1')
    embedder.cluster(method='dbscan', eps=0.6, min_samples=3, colnameoflabel='test2')
    print(embedder.metadata)


    # Example usage 6: Assigning cluster labels to new embeddings
    new_strings = [
        "Atherosclerosis is a condition where plaque builds up inside the arteries, leading to narrowing and hardening of the arteries.",
        "Insulin resistance occurs when cells in the body don't respond properly to insulin, leading to high blood sugar levels.",
        "Bronchitis is an inflammation of the lining of the bronchial tubes, which carry air to and from the lungs."
    ]
    assigned_labels = embedder.assign_labels_to_new_embeddings(new_strings, 'test1', method='knn')
    print("Assigned Labels:", assigned_labels)   


    # Example usage 7: Matching queries
    query = "A cardiovascular condition characterized by chest pain and discomfort."
    matched_items = embedder.match(query, topk=2)
    print(matched_items)

    queries = [
        "A chronic respiratory disease that makes it difficult to breathe.",
        "A neurological condition that causes memory loss and cognitive decline.",
        "An endocrine disorder resulting in elevated blood glucose levels."
    ]
    matched_results = embedder.match(queries, threshold=0.7, return_sims=True)
    print(matched_results)
