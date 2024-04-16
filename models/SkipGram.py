import torch.nn as nn

class SkipGram(nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size, embed_dimension):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dimension,
            max_norm=embed_dimension,
        )
        self.linear = nn.Linear(
            in_features=embed_dimension,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x