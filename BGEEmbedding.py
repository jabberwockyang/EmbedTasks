from __future__ import annotations
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
import torch
import numpy as np
import os
from tqdm import tqdm
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

class embeddingDataset(Dataset):
    def __init__(self, modelname, descriptions, s2p: bool =False):
        self.descriptions = descriptions
        self.tokenizer = AutoTokenizer.from_pretrained(modelname, local_files_only=True )
        self.s2p = s2p

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        instruction = "为这个句子生成表示以用于检索相关文章："

        if self.s2p == False:
            encoded_input = self.tokenizer(self.descriptions[idx], 
                                           padding=False, truncation=True,max_length=512)
        else:
            # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
            instrQueies = [instruction + q for q in self.descriptions]
            encoded_input = self.tokenizer(instrQueies[idx], 
                                           padding=False, truncation=True,max_length=512)
        return encoded_input




class BGEEmbedding():
    def __init__(self,lang,local_files_only = True):
        self.local_files_only = local_files_only
        if lang == 'zh':
            self.modelname = 'BAAI/bge-large-zh-v1.5'
        elif lang == 'en':
            self.modelname = 'BAAI/bge-large-en-v1.5'
        else:
            raise ValueError('language not supported')
        pass

    def _my_collate_fn(self,batch):
        tokenizer = AutoTokenizer.from_pretrained(self.modelname, local_files_only=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest', return_tensors='pt')
        return data_collator(batch)

    def generate_embeddings(self, queries, s2p: bool =False,batch_size: int = 8):
        '''
        generate embeddings for queries
        input: 
        - queries: list of queries
        - s2p: bool, if True, add an instruction to query  add for s2p(short query to long passage) retrieval task (not add instruction for passages) 
        - batch_size: int, batch size for generating embeddings
        output:
        - sentence_embeddings: tensor, nXd n: number of sentences, d: dimension of embeddings
        
        '''

        dataset = embeddingDataset(self.modelname, queries, s2p=s2p)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn = self._my_collate_fn)
        
        model = AutoModel.from_pretrained(self.modelname, local_files_only= self.local_files_only)
        model.eval()

        embeddings_list = []
        # Compute token embeddings
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                model_output = model(**batch) 
                # Perform pooling. In this case, cls pooling.
                sentence_embeddings = model_output[0][:, 0] 
                normalized_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                embeddings_list.append(normalized_embeddings.cpu().numpy())
                
        return np.vstack(embeddings_list)


    def rerank(self, pairs):
        '''
        rerank the pairs of queries and passages
        input:
        - pairs: list of pairs of queries and passages
        output:
        - scores: tensor, nX1, n: number of pairs
        '''

        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base', local_files_only=self.local_files_only)
        model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base', local_files_only=self.local_files_only)
        model.eval()

        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        
        return scores.tolist()  


if __name__ == "__main__":
    queries = ['what is panda?', 'how to make a cake?']
    embedder = BGEEmbedding()
    embeddings = embedder.generate_embeddings(queries=queries, s2p=False)
    print(embeddings)
    print(embeddings.shape)
    pairs = [['what is panda?', 'hi'], 
            ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
    scores = embedder.rerank(pairs)
    print(scores)
    