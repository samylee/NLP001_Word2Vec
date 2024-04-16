import torch.nn as nn


class CBOW(nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size, embed_dimension):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dimension,
            max_norm=1.0,
        )
        self.linear = nn.Linear(
            in_features=embed_dimension,
            out_features=vocab_size,
        )

    def forward(self, x):
        x = self.embeddings(x)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x