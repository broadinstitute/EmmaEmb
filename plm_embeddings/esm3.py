import logging
import numpy as np
import os

import torch
from esm.models.esm3 import ESM3
from esm.models.esmc import ESMC
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.sdk.api import ESMProtein, SamplingConfig, LogitsConfig
from torch.utils.data import DataLoader, Dataset

from plm_embeddings.embedding_handler import EmbeddingHandler
from huggingface_hub import login


HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")


class ProteinDataset(Dataset):
    def __init__(self, sequences_dict):
        """
        Initialize with a dictionary of {protein_id: sequence}.
        """
        self.protein_ids = list(sequences_dict.keys())
        self.sequences = list(sequences_dict.values())

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.protein_ids[idx], self.sequences[idx]
    

# Script kindly provided by Evolutionary Scale
# https://github.com/evolutionaryscale/esm
class Esm3(EmbeddingHandler):
    def __init__(
        self,
        logger: logging.Logger = None,
        no_gpu: bool = False,
    ):
        super().__init__(logger, no_gpu)

    def get_embedding(self, 
                      model_id: str,
                      protein_sequences: dict,
                      output_dir: str,
                      truncation_seq_length: int,
                      batch_size: int=16,
                      model_dir: str=None,
                      layer: int = None,
                      per_protein: bool = True,
                      ):
        
        if layer:
            raise ValueError("ESM3 embeddings only \
                implemented for the last layer.")
        # TODO: parameterise model_id
        login(token=HF_AUTH_TOKEN)
        if model_id == "esm3-sm-open-v1":
            client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=self.device)
        elif model_id == "esmc-300m-2024-12":
            client = ESMC.from_pretrained("esmc_300m").to(self.device)
        dataset = ProteinDataset(protein_sequences)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        protein_ids = []
        embeddings_list = []
        
        for batch in dataloader:
            batch_protein_ids, batch_sequences = batch
            proteins = [ESMProtein(sequence=seq) for seq in batch_sequences]
            
            for pid, protein in zip(batch_protein_ids, proteins):
                output_file = output_dir / f"{pid}.npy"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                protein_tensor = client.encode(protein)
                
                if model_id == "esm3-sm-open-v1":
                    output = client.forward_and_sample(
                        protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
                    )
                    if per_protein:
                        embedding = output.per_residue_embedding.mean(dim=0).cpu().numpy()
                    else:
                        # strip BOS/EOS tokens, keep per-residue embeddings
                        embedding = output.per_residue_embedding[1:-1].cpu().numpy()
                elif model_id == "esmc-300m-2024-12":
                    logits_output = client.logits(
                        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
                    )
                    embeddings = logits_output.embeddings[0]

                    if per_protein:
                        embedding = embeddings.mean(dim=0).cpu().numpy()
                    else:
                        valid_embeddings = embeddings[1:-1]    
                        embedding = valid_embeddings.cpu().numpy()
                
                np.save(
                        output_file,
                        embedding,
                    )
                print()