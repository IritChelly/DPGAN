from typing_extensions import final
import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device, temperature=0.5, verbose=False):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
        self.verbose = verbose
        self.device = device
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2).to(self.device)
        if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        if self.verbose: print(f"sim_ij:", sim_ij, "\n")

        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask.to(self.device) * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        if self.verbose: print("Contrastive loss:", loss, "\n")
        return loss


# I = torch.tensor([[1.0, 2.0], [3.0, -2.0], [1.0, 5.0]])
# J = torch.tensor([[1.0, 0.75], [2.8, -1.75], [1.0, 4.7]])
# loss = ContrastiveLoss(batch_size=3, temperature=1.0, verbose=True)
# loss(I, J)