import torch
from transformers import BioGptTokenizer, BioGptForCausalLM
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
import numpy as np
import pandas as pd
import time

class EmbeddingDataset(Dataset):
    def __init__(self, descriptions):
        self.descriptions = descriptions
        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt", local_files_only=True)

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        encoded_input = self.tokenizer(self.descriptions[idx],
                                    padding=False,
                                    truncation=False)
        return encoded_input
    
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.hidden_states[-1]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def my_collate_fn(batch):
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt", local_files_only=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,return_tensors="pt")
    return data_collator(batch)

class BioGptEmbedding():
    def __init__(self,local_files_only=True):
        self.local_files_only = local_files_only
        pass

    def generate_embeddings(self, descriptions,batch_size):
        dataset = EmbeddingDataset(descriptions)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn = my_collate_fn)
        
        model = BioGptForCausalLM.from_pretrained("microsoft/biogpt", local_files_only = self.local_files_only)
        model.eval()
        
        embeddings = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].squeeze(1)
                attention_mask = batch['attention_mask'].squeeze(1)
                
                model_output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                batch_embeddings = mean_pooling(model_output, attention_mask)
                embeddings.extend(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

# Example usage
if __name__ == "__main__":
    # Load your ICD-10-CM descriptions
    # This is just an example, replace it with loading your actual data
    
    df = pd.read_csv('Data/ICD10/icd10cm_codes_2022.csv')
    icd10_descriptions = df['name'][:1000].tolist()
    for batch_size in [200]:
        t1 = time.time()
        embeddings = BioGptEmbedding.generate_embeddings(icd10_descriptions,batch_size)
        t2 = time.time()
        print("Generated Embeddings Shape:", embeddings.shape,"Batch Size:",batch_size,"Time taken:", t2-t1)
        
    
