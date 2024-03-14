from embed_task import embed_task_with_repo

import pandas as pd
import random

df = pd.read_csv('/mnt/yangyijun/embeddingtools/data/ambossdf.csv')

trainlen = int(df.shape[0] * 0.8)
trainindex = random.sample(range(df.shape[0]), trainlen)
train = df.iloc[trainindex]
with open('trainindex.txt', 'w') as f:
    for item in trainindex:
        f.write("%s\n" % item)


embedder = embed_task_with_repo('bge-en', train,colname= 'string')
# embedder.load_embeddings("embeddings_df.npy")
embedder.generate_embeddings_for_repo(batch_size=8)
# embedder.get_similarity_distribution(plot=True)

embedder.save_embeddings("embeddings_df.npy")