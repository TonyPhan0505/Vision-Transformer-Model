import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbeddings(nn.Module):
    """Calculates patch embedding
    of shape `(batch_size, seq_length, hidden_size)`.
    """
    # num_channels = 3 for RGB, 1 for Grayscale
    def __init__(self, image_size: int, patch_size: int, hidden_size: int, num_channels: int = 3):
        super().__init__()
        # #########################
        # Finish Your Code HERE
        # #########################

        self.projection = nn.Linear(num_channels * patch_size * patch_size, hidden_size)
        num_patches = (image_size // patch_size) ** 2
        
        # #########################

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = x.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        
        # #########################
        # Finish Your Code HERE
        # #########################

        # Calculate Patch Embeddings, then flatten into
        # batched 1D sequence (batch_size, seq_length, hidden size)
        x = x.view(batch_size, self.num_channels, height // self.patch_size, self.patch_size, width // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.flatten(1, 2) # [batch_size, num of patches, self.num_channels, self.patch_size, self.patch_size]
        x = x.flatten(2, 4) # [batch_size, seq_length, hidden size]
        embeddings = self.projection(x)
        # #########################
        return embeddings

class PositionEmbedding(nn.Module):
    def __init__(self, num_patches, hidden_size):
        """Given patch embeddings, 
        calculate position embeddings with [CLS] and [POS].
        """
        super().__init__()
        # #########################
        # Finish Your Code HERE
        # #########################

        self.hidden_size = hidden_size
        self.num_patches = num_patches
        
        # Specify [CLS] and positional embedding as learnable parameters

        self.cls_token = nn.Parameter(torch.rand(1, hidden_size))
        self.position_embeddings = nn.Parameter(torch.randn(1, 1 + self.num_patches, self.hidden_size))
        self.position_embeddings.requires_grad = False

        # #########################

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # #########################
        # Finish Your Code HERE
        # #########################

        # Concatenate [CLS] token with embedded patch tokens
        embeddings = torch.stack([torch.vstack((self.cls_token, embeddings[i])) for i in range(len(embeddings))])

        # Then add positional encoding to each token
        embeddings = embeddings + self.position_embeddings[:,:self.num_patches + 1]

        # #########################
        return embeddings

class GELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    """TODO: (0.25 out of 8) Residual Attention Block.
    """
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        # #########################
        # Finish Your Code HERE
        # #########################

        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)    # Refer to nn.MultiheadAttention
        self.ln_2 = nn.LayerNorm(d_model)
        # A trick is to use nn.Sequential to specify multiple layers at once
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), 
            nn.GELU(), 
            nn.Dropout(0.2),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(0.2)
        )
        # #########################

    def forward(self, x: torch.Tensor):

        # #########################
        # Finish Your Code HERE
        # #########################

        # LayerNorm -> Multi-head Attention
        inp_x_1 = self.ln_1(x)
        # Residual connection against x
        x = x + self.attn(inp_x_1, inp_x_1, inp_x_1)[0]
        # LayerNorm -> MLP Block
        inp_x_2 = self.ln_2(x)
        # Residual connection against x
        out = x + self.mlp(inp_x_2)
        # #########################

        return out

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2)  # (batch_size, seqlen, dim) -> (seqlen, batch_size, dim)
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)  # (seqlen, batch_size, dim) -> (batch_size, seqlen, dim)
        return x

class ViT(nn.Module):
    """Vision Transformer.
    """
    def __init__(self, image_size: int, patch_size: int, num_channels: int, hidden_size: int, layers: int, heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        # #########################
        # Finish Your Code HERE
        # #########################
        self.patch_embed = PatchEmbeddings(image_size, patch_size, self.hidden_size, num_channels)

        self.pos_embed = PositionEmbedding(self.patch_embed.num_patches, self.hidden_size)

        self.ln_pre = nn.LayerNorm(self.hidden_size)

        self.transformer = Transformer(self.hidden_size, layers, heads)     # TODO: Use the provided transformer codeblock

        self.ln_post = nn.LayerNorm(self.hidden_size)

        # #########################


    def forward(self, x: torch.Tensor):
        # #########################
        # Finish Your Code HERE
        # #########################
        x = self.patch_embed(x)
        x = self.ln_pre(x)
        x = self.pos_embed(x)
        # #########################
        x = self.transformer(x)
        out = self.ln_post(x)
        out = out[:, 0, :]
        return out

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int = 10):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self, 
        feats: torch.Tensor,
        ) -> torch.Tensor:
        out = self.classifier(feats)
        return out

class LinearEmbeddingHead(nn.Module):
    """Given features from ViT, generate linear embedding vectors.
    """
    def __init__(self, hidden_size: int, embed_size: int = 64):
        super().__init__()
        self.embed_size = embed_size
        # #########################
        # Finish Your Code HERE
        # #########################
        self.projection = nn.Linear(hidden_size, embed_size)
        # #########################

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # #########################
        # Finish Your Code HERE
        # #########################
        out = self.projection(feats)
        # #########################
        return out
