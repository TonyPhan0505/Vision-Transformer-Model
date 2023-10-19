import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbeddings(nn.Module):
    """TODO: (0.5 out of 8) Calculates patch embedding
    of shape `(batch_size, seq_length, hidden_size)`.
    """
    def __init__(
            self, 
            image_size: int,
            patch_size: int,
            hidden_size: int,
            num_channels: int = 3,      # 3 for RGB, 1 for Grayscale
        ):
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
    
    def getNumPatches(self):
        return self.num_patches

    def forward(
            self, 
            x: torch.Tensor,
        ) -> torch.Tensor:
        batch_size, num_channels, height, width = x.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        
        # #########################
        # Finish Your Code HERE
        # #########################

        # Calculate Patch Embeddings, then flatten into
        # batched 1D sequence (batch_size, seq_length, hidden_size)
        x = x.view(batch_size, num_channels, height // self.patch_size, self.patch_size, width // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(batch_size, -1, num_channels * self.patch_size * self.patch_size)
        embeddings = self.projection(x)
        # #########################
        return embeddings

class PositionEmbedding(nn.Module):
    def __init__(
            self,
            num_patches,
            hidden_size,
        ):
        """TODO: (0.5 out of 8) Given patch embeddings, 
        calculate position embeddings with [CLS] and [POS].
        """
        super().__init__()
        # #########################
        # Finish Your Code HERE
        # #########################
        
        # Specify [CLS] and positional embedding as learnable parameters

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))

        # #########################

    def forward(
            self,
            embeddings: torch.Tensor
        ) -> torch.Tensor:
        # #########################
        # Finish Your Code HERE
        # #########################

        # Concatenate [CLS] token with embedded patch tokens
        cls_token = self.cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat([cls_token, embeddings], dim=1)
        # Then add positional encoding to each token
        if embeddings.size(1) > self.position_embeddings.size(1):
            embeddings = embeddings[:, :self.position_embeddings.size(1), :]
        embeddings += self.position_embeddings
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

        self.attn = nn.MultiheadAttention(d_model, n_head)    # Refer to nn.MultiheadAttention
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model)
        )     # A trick is to use nn.Sequential to specify multiple layers at once
        self.ln_2 = nn.LayerNorm(d_model)

        # #########################

    def forward(self, x: torch.Tensor):

        # #########################
        # Finish Your Code HERE
        # #########################

        # LayerNorm -> Multi-head Attention
        inp_x = self.ln_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        # Residual connection against x
        # LayerNorm -> MLP Block
        # Residual connection against x
        out = x + self.mlp(self.ln_2(x))
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
    """TODO: (0.5 out of 8) Vision Transformer.
    """
    def __init__(
            self, 
            image_size: int, 
            patch_size: int, 
            num_channels: int,
            hidden_size: int, 
            layers: int, 
            heads: int,
        ):
        super().__init__()
        self.hidden_size = hidden_size
        # #########################
        # Finish Your Code HERE
        # #########################
        self.patch_embed = PatchEmbeddings(image_size, patch_size, hidden_size, num_channels)
        num_patches = self.patch_embed.getNumPatches()
        self.pos_embed = PositionEmbedding(num_patches, hidden_size)

        self.ln_pre = nn.LayerNorm(hidden_size)

        self.transformer = Transformer(hidden_size, layers, heads)  # TODO: Use the provided transformer codeblock

        self.ln_post = nn.LayerNorm(hidden_size)
            
        # #########################


    def forward(self, x: torch.Tensor):
        # #########################
        # Finish Your Code HERE
        # #########################
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.ln_pre(x)
        x = self.transformer(x)
        out = self.ln_post(x)
        out = out[:, 0, :]
        # #########################
        return out

class ClassificationHead(nn.Module):
    def __init__(
            self, 
            hidden_size: int,
            num_classes: int = 10,
        ):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
            self, 
            feats: torch.Tensor,
        ) -> torch.Tensor:
        out = self.classifier(feats)
        return out

class LinearEmbeddingHead(nn.Module):
    """TODO: (0.25 out of 8) Given features from ViT, generate linear embedding vectors.
    """
    def __init__(
            self, 
            hidden_size: int,
            embed_size: int = 64,
        ):
        super().__init__()
        self.embed_size = embed_size
        # #########################
        # Finish Your Code HERE
        # #########################
        self.projection = nn.Linear(hidden_size, embed_size)
        # #########################

    def forward(
        self, 
        feats: torch.Tensor,
        ) -> torch.Tensor:
        # #########################
        # Finish Your Code HERE
        # #########################
        out = self.projection(feats)
        # #########################
        return out