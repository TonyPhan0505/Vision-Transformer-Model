import torch

from vit_model import *

IMAGE_SIZE = 28
PATCH_SIZE = 7
HIDDEN_SIZE = 64
BATCH_SIZE = 8
LAYERS = 3
HEADS = 4
IN_CHANNELS = 3

if __name__ == "__main__":
    # Fake random tensor to check inputs
    x = torch.rand((BATCH_SIZE, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))

    # Check PatchEmbeddings
    # -----
    patch_embed_layer = PatchEmbeddings(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        hidden_size=HIDDEN_SIZE,
    )
    embeds = patch_embed_layer(x)
    assert embeds.shape == (BATCH_SIZE, 16, HIDDEN_SIZE)

    # Check Position Embeddings
    # -----
    pos_embed_layer = PositionEmbedding(
        num_patches=patch_embed_layer.num_patches, 
        hidden_size=HIDDEN_SIZE,
        )

    x = pos_embed_layer(embeds)
    assert x.shape == (BATCH_SIZE, 17, HIDDEN_SIZE)
    
    # Check ViT
    # -----
    model = ViT(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_channels=IN_CHANNELS, hidden_size=HIDDEN_SIZE, layers=LAYERS, heads=HEADS)
    x = torch.rand((BATCH_SIZE, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    feats = model(x)
    assert feats.shape == (BATCH_SIZE, HIDDEN_SIZE)

    # Check ViTClassificationHead
    # -----
    model_classifier = ClassificationHead(hidden_size=HIDDEN_SIZE, num_classes=10)
    out = model_classifier(feats)
    assert out.shape == (BATCH_SIZE, 10)

    # Check ViTLinearEmbeddingHead
    # -----
    model_embedding = LinearEmbeddingHead(hidden_size=HIDDEN_SIZE, embed_size=64)
    out = model_embedding(feats)
    assert out.shape == (BATCH_SIZE, 64)
