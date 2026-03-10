# %% [markdown]
# 
# %% [markdown]
# # Self-Supervised Image Representation Learning using Masked Autoencoders (MAE)
# ## **GenAI Assignment 02 — Spring 2026**
# ## This notebook implements a Masked Autoencoder (MAE) from scratch using base PyTorch.
# ## Architecture: ViT-Base encoder + ViT-Small decoder, trained on TinyImageNet.
# %% [markdown]
# ## Part 0: Environment Setup & Data Loading
# %% [markdown]
# # Cell 0.1 — Imports & Device Setup

# %%
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Device: {device} | GPUs available: {num_gpus}")


# %% [markdown]
# # Cell 0.2 — Hyperparameters

# %%
# First, we define the path to our dataset. 
# Kaggle usually mounts datasets under /kaggle/input/
# The code below automatically searches for the directory containing 'train' and 'val' folders
# so that it works regardless of how Kaggle exactly unzips the dataset.
import os
DATASET_PATH = "/kaggle/input/tiny-imagenet/tiny-imagenet-200" # Default fallback
for root, dirs, files in os.walk('/kaggle/input'):
    if 'train' in dirs and 'val' in dirs:
        DATASET_PATH = root
        break
print(f"Using dataset path: {DATASET_PATH}")

# ==================== ARCHITECTURE HYPERPARAMETERS ====================
IMAGE_SIZE = 224      # Standard image size for Vision Transformers (ViT)
PATCH_SIZE = 16       # We split the 224x224 image into 16x16 pixel blocks (patches)
                      # So, 224 / 16 = 14 patches per row/col -> 14 * 14 = 196 total patches
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 196 patches in total

# ==================== MAE SPECIFIC HYPERPARAMETERS ====================
MASK_RATIO = 0.75     # The core idea of MAE: we hide 75% of the patches!
                      # This forces the model to heavily learn the global structure of images
                      # rather than just copying adjacent pixels.
                      
# Calculate exactly how many patches we keep (visible) vs how many we hide (masked)
NUM_VISIBLE = int(NUM_PATCHES * (1 - MASK_RATIO))  # 196 * 0.25 = 49 visible patches
NUM_MASKED = NUM_PATCHES - NUM_VISIBLE             # 196 - 49 = 147 masked patches

# ==================== TRAINING HYPERPARAMETERS ====================
BATCH_SIZE = 48       # Number of images processed at once. Set to 48 to maximize the 15GB VRAM on T4 GPUs
NUM_EPOCHS = 50       # How many times we pass over the entire dataset
LEARNING_RATE = 1.5e-4# The base step size for AdamW optimizer
WEIGHT_DECAY = 0.05   # Regularization to prevent overfitting by penalizing large weights
WARMUP_EPOCHS = 5     # We'll gradually increase the learning rate for the first 5 epochs (scheduler)

print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
print(f"Total patches: {NUM_PATCHES}, Visible: {NUM_VISIBLE}, Masked: {NUM_MASKED}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")

# %% [markdown]
# # Cell 0.3 — Dataset Class & DataLoaders
# 

# %%
# === Custom TinyImageNet Dataset ===

from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Inverse normalization for visualization
inv_normalize = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.],
                         std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                         std=[1., 1., 1.]),
])


class TinyImageNetDataset(Dataset):
    """Custom TinyImageNet dataset loader."""

    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []

        if split == 'train':
            train_dir = self.root_dir / 'train'
            for class_dir in sorted(train_dir.iterdir()):
                images_dir = class_dir / 'images'
                if images_dir.exists():
                    for img_path in sorted(images_dir.glob('*.JPEG')):
                        self.image_paths.append(str(img_path))
        elif split == 'val':
            val_dir = self.root_dir / 'val'
            val_images_dir = val_dir / 'images'
            if val_images_dir.exists():
                for img_path in sorted(val_images_dir.glob('*.JPEG')):
                    self.image_paths.append(str(img_path))

        print(f"Loaded {len(self.image_paths)} images for '{split}' split.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


# Create datasets and dataloaders
train_dataset = TinyImageNetDataset(DATASET_PATH, split='train', transform=train_transform)
val_dataset = TinyImageNetDataset(DATASET_PATH, split='val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")


# %% [markdown]
# # Cell 0.4 — Dataset Verification (Sample Visualization)
# 

# %%
# Verify dataset with a sample
sample = train_dataset[0]
print(f"Sample shape: {sample.shape}")  # Should be [3, 224, 224]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i in range(4):
    img = inv_normalize(train_dataset[i])
    img = img.permute(1, 2, 0).clamp(0, 1).numpy()
    axes[i].imshow(img)
    axes[i].set_title(f"Sample {i+1}")
    axes[i].axis('off')
plt.suptitle("TinyImageNet Samples (Resized to 224×224)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 1: Patchification & Masking
# %% [markdown]
# # Cell 1.1 — PatchEmbed, patchify, unpatchify, random_masking
# 

# %%
# ==============================================================================
# PART 1: PATCHIFICATION AND MASKING
# This is where the core logic of Masked Autoencoders lives.
# ==============================================================================

class PatchEmbed(nn.Module):
    """
    Step 1: Patch Embedding
    Vision Transformers don't process pixels directly; they process 'patches' or 'tokens'.
    This class takes the (3, 224, 224) image and flattens it into a sequence of (196, 768) embeddings.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 196 patches
        
        # TRICK: Instead of manually slicing the image into patches and running a Linear layer on each,
        # we can use a 2D Convolution where the kernel size AND the stride exactly equal the patch size (16).
        # This extracts non-overlapping 16x16 blocks and projects their 16*16*3=768 pixels into 'embed_dim' features simultaneously!
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x input shape: (B, 3, 224, 224)
        x = self.proj(x) 
        # After Conv2d: shape is (B, embed_dim, 14, 14) -> 14x14 grid of patches, each represented by a 768-dim vector
        
        # We need a 1D sequence for the Transformer, so we flatten the 14x14 spatial dimensions.
        x = x.flatten(2).transpose(1, 2)
        # Final shape: (B, 196, embed_dim) -> representing 196 tokens, just like words in a sentence.
        return x


def patchify(imgs, patch_size=16):
    """
    Utility function used during LOSS CALCULATION.
    While PatchEmbed converts images to 'latent features', this function converts images directly into 'pixel patches'.
    We need this because the MAE loss is computed by comparing the *predicted pixels* against the *actual original pixels*.
    """
    B, C, H, W = imgs.shape
    h = w = H // patch_size
    
    # 1. Reshape image into grid of patches: (Batch, Channels, 14 rows, 16px, 14 cols, 16px)
    x = imgs.reshape(B, C, h, patch_size, w, patch_size)
    
    # 2. Swap dimensions so spatial grid is together: (Batch, 14 rows, 14 cols, 16px, 16px, Channels)
    x = x.permute(0, 2, 4, 3, 5, 1)
    
    # 3. Flatten the rows/cols into 196 patches, and the pixels into 16*16*3 = 768 values per patch.
    # Output shape: (B, 196, 768)
    x = x.reshape(B, h * w, patch_size * patch_size * C)
    return x


def unpatchify(patches, patch_size=16, img_size=224):
    """
    Utility function used during VISUALIZATION.
    Takes the model's output (which is shape B, 196, 768 vector predictions) and reformats it 
    back into a standard image shape (B, 3, 224, 224) so we can plot it with matplotlib.
    This does the exact reverse operations of patchify().
    """
    B = patches.shape[0]
    h = w = img_size // patch_size
    C = 3
    
    # 1. Unflatten: (B, 196 patches, 768 pixels) -> (B, 14, 14, 16, 16, 3)
    x = patches.reshape(B, h, w, patch_size, patch_size, C)
    
    # 2. Permute back to image structure: (B, 3, 14, 16, 14, 16)
    x = x.permute(0, 5, 1, 3, 2, 4)
    
    # 3. Reshape spatial dims: (B, 3, 224, 224)
    x = x.reshape(B, C, img_size, img_size)
    return x


def random_masking(x, mask_ratio=0.75):
    """
    Step 2: Generate masks.
    This is the SECRET SAUCE of Masked Autoencoders.
    Takes 196 patches, securely shuffles them, and completely throws away 75% of them.
    """
    B, N, D = x.shape  # Batch size, Number of patches (196), Embedding dim (768)
    num_keep = int(N * (1 - mask_ratio))  # How many patches to keep (49)

    # To mask randomly, we generate a random noise matrix of shape (B, 196).
    # We assign a random number to every single patch.
    noise = torch.rand(B, N, device=x.device)

    # `argsort` tells us the indices that would sort the noise matrix.
    # Essentially, this gives us a random shuffled list of indices from 0 to 195!
    ids_shuffle = torch.argsort(noise, dim=1)
    
    # We also need to remember how to UNSHUFFLE later (so the decoder can put patches back in their correct spatial structure).
    # Sorting the shuffled indices gives us the map back to the original order.
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Take the first 49 indices from our randomly shuffled list. These are our "visible" patches.
    ids_keep = ids_shuffle[:, :num_keep]

    # Use torch.gather to physically extract ONLY the 49 keeping patches from 'x'.
    # x_visible is now shape (B, 49, 768). The other 147 patches are completely dropped to save memory/compute!
    x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

    # Finally, generate a binary mask for loss calculation (0 = visible, 1 = masked).
    mask = torch.ones(B, N, device=x.device)
    mask[:, :num_keep] = 0 # First num_keep patches are '0' (visible in our shuffled state)
    
    # Unshuffle the mask so the 0s and 1s align with the actual image grid layout.
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_visible, mask, ids_restore, ids_keep

# Quick tests ensure the shapes align. 196 goes in, 49 comes out.

# %% [markdown]
# ## Part 2: Transformer Building Blocks
# %% [markdown]
# # Cell 2.1 — MultiHeadSelfAttention, MLP, TransformerBlock
# 

# %%
# ==============================================================================
# PART 2: TRANSFORMER BUILDING BLOCKS
# These are the standard components of a Vision Transformer (ViT).
# They are used by both the Encoder and the Decoder to process sequences of patches.
# ==============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA).
    The "Self-Attention" mechanism allows each patch to look at every other patch in the image 
    and decide which patches are most important for understanding its own content.
    """
    def __init__(self, embed_dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Dimension of each individual attention head
        self.scale = self.head_dim ** -0.5      # Scaling factor to prevent dot products from getting too large

        # A single linear layer that computes Queries (Q), Keys (K), and Values (V) all at once
        # Output is exactly 3 times the embedding dimension.
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Final projection to combine the multi-head results back together
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # Batch size, Number of tokens, Channels (embed_dim)
        
        # 1. Compute Q, K, V for all heads simultaneously
        # Reshape to separate the 'num_heads' and 'head_dim'
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # Unbind splits the tensor along dimension 0 into Q, K, and V

        # 2. Compute the Attention Matrix
        # (Q dot K^T) / sqrt(d_k)
        # This gives an N x N matrix representing how much each token attends to every other token.
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Softmax turns the raw scores into probabilities that sum to 1.0 per row.
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 3. Apply the Attention weights to the Values
        # (Attention Matrix) dot V
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # Reassemble the heads: (B, N, C)
        
        # 4. Final output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) / Feed-Forward Network.
    After patches look at each other (Self-Attention), they process what they observed individually 
    through this two-layer neural network.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        # Usually, transformer MLPs expand the dimensionality by 4x internally, then project back.
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()  # GELU is standard in ViT instead of ReLU
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single complete Transformer block combining Attention and MLP layers.
    It uses the Pre-LayerNorm architecture (Norm before Attention/MLP, rather than after).
    This has been mathematically proven to greatly improve training stability for deep networks like MAE.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0,
                 qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        
        # Component 1: Self-Attention module
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, qkv_bias=qkv_bias,
                                            attn_drop=attn_drop, proj_drop=drop)
                                            
        # Component 2: Feed-Forward MLP
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio) # 4x expansion factor
        self.mlp = MLP(embed_dim, hidden_features=mlp_hidden, drop=drop)

    def forward(self, x):
        # 1. Residual Connection around Attention (Input + Attention(Norm(Input)))
        x = x + self.attn(self.norm1(x))
        # 2. Residual Connection around MLP (Input + MLP(Norm(Input)))
        x = x + self.mlp(self.norm2(x))
        
        # Residual connections ensure gradients flow freely backwards even in 12 or 24 layer networks.
        return x

# %% [markdown]
# ## Part 3: ViT Encoder (ViT-Base B/16)
# %% [markdown]
# #  Cell 3.1 — ViTEncoder Class
# 

# %%
# ==============================================================================
# PART 3: ViT ENCODER
# The Encoder's job is strictly to process the 49 VISIBLE patches into rich abstract representations.
# ==============================================================================

class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder (Based on ViT-Base architecture).
    
    Key Features of the MAE Encoder:
    1. It ONLY receives the 49 unmasked (visible) patches. This makes it 4x faster and uses 4x less memory!
    2. It does NOT process 'mask tokens'. Mask tokens are completely ignored here.
    3. Because patches are unordered sequence elements to a Transformer, we MUST add Positional Embeddings 
       so the model knows 'where' in the original image these 49 patches came from.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        
        # 1. The Patch Embedder (turns image into tokens)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches  # 196
        self.embed_dim = embed_dim

        # 2. Positional Embeddings (Shape: 1, 196, 768)
        # Learnable parameter that stores the "location signature" for all 196 possible grid positions.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # 3. The Core Transformer engine
        # ViT-Base uses 12 stacked transformer blocks. 
        # Deep models build hierarchical features: early layers detect edges, deep layers detect objects.
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio,
                             qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        
        # 4. Final normalization layer to stabilize the output before handing off to the decoder
        self.norm = nn.LayerNorm(embed_dim)

        self._init_pos_embed()
        self.apply(self._init_weights)

    def _init_pos_embed(self):
        # We start with sinusoidal positional embeddings (like the original Transformer paper)
        # It provides a strong prior that adjacent tokens are spatially correlated.
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _init_weights(self, m):
        # Best practice weight initialization: Xavier Uniform for linear layers
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_keep):
        """
        The actual execution flow of the Encoder.
        x: Images (Batch, 3, 224, 224)
        """
        # Step 1: Turn (B, 3, 224, 224) pixels into (B, 196, 768) patch tokens
        x = self.patch_embed(x) 

        # Step 2: Add Positional Embeddings IMMEDIATELY
        # We must add them to ALL 196 patches BEFORE we mask them, so the visible patches remember their original coordinates!
        x = x + self.pos_embed

        # Step 3: Run the Random Masking
        # This completely drops 147 patches.
        # x becomes 'x_visible' of shape (B, 49, 768).
        x, mask, ids_restore, ids_keep = random_masking(x, MASK_RATIO)

        # Step 4: Process the 49 patches through 12 heavy Transformer blocks!
        # Because N=49 instead of 196, the Attention mechanism (O(N^2)) is mathematically (196/49)^2 = 16x cheaper!
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        
        # Return the encoded representations of the 49 patches, plus the structural data (mask, ids_restore) 
        # so the Decoder knows exactly where to put them back.
        return x, mask, ids_restore

# %% [markdown]
# ## Part 4: ViT Decoder (ViT-Small S/16)
# 
# %% [markdown]
# # Cell 4.1 — ViTDecoder Class
# 

# %%
# ==============================================================================
# PART 4: ViT DECODER
# The Decoder's job is to take the 49 encoded patches + 147 empty "Mask Tokens", 
# figure out their order, and hallucinate/reconstruct the missing pixels!
# ==============================================================================

class ViTDecoder(nn.Module):
    """
    Vision Transformer Decoder (Based on ViT-Small architecture).
    
    Key Features:
    1. It is ASYMMETRIC. The encoder is huge (Base) but the Decoder is small.
       This is because reconstructing pixels from rich features is an "easier" task requiring less logic
       than actually understanding the image. This saves massive compute.
    2. Input to decoder is ALL 196 tokens (49 encoded + 147 mask tokens).
    """
    def __init__(self, num_patches=196, patch_size=16,
                 encoder_embed_dim=768, decoder_embed_dim=384,
                 depth=12, num_heads=6, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_patches = num_patches
        self.decoder_embed_dim = decoder_embed_dim

        # 1. Dimension translation: Encoder outputs 768-dim, but Decoder only uses 384-dim (Small).
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        # 2. THE MASK TOKEN
        # This is a single, learnable vector of size 384. 
        # We will dynamically copy/paste this vector 147 times to represent every missing patch!
        # The model "learns" during training what a generic placeholder should look like.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # 3. Positional Embeddings for the Decoder
        # Wait, didn't the encoder have positional embeddings?
        # Yes, but we just added 147 brand new mask tokens that don't know where they belong!
        # So we MUST add a fresh set of positional embeddings to all 196 tokens so the mask tokens know what to predict.
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))

        # 4. Decoder Transformer blocks (Lighter than encoder: only 6 heads, 384 dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, num_heads, mlp_ratio,
                             qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_embed_dim)

        # 5. Prediction Head (The final output)
        # Projects the 384-dim latent token back into 768 RAW PIXEL VALUES (16px * 16px * 3 channels)
        self.pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * 3, bias=True)

        self._init_pos_embed()
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_pos_embed(self):
        pos_embed = get_2d_sincos_pos_embed(self.decoder_embed_dim, int(self.num_patches ** 0.5))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore):
        """
        x: (B, 49, 768) - The processed visible tokens from the encoder
        ids_restore: (B, 196) - Map of where the 49 tokens actually belong in the 196 grid.
        """
        # Step 1: Compress 768-dim encoder tokens down to 384-dim decoder tokens
        x = self.decoder_embed(x) 

        # Step 2: Create the mask tokens
        B, N_vis, D = x.shape  # B, 49, 384
        N = self.num_patches   # 196
        # Expand the single mask parameter into a massive block of 147 mask tokens per image
        mask_tokens = self.mask_token.expand(B, N - N_vis, -1)  # (B, 147, 384)
        
        # Step 3: Concatenate visible patches and mask patches
        # Right now they are unordered: all 49 visible tokens are at the beginning, followed by 147 masks.
        x_ = torch.cat([x, mask_tokens], dim=1)  # (B, 196, 384)

        # Step 4: UNSHUFFLE
        # Using the `ids_restore` mapping, we rearrange the 196 tokens so they exactly match the spatial grid of the image!
        # Beautifully, the 49 real patches slide right into their original grid spots, and the empty spots are filled by mask tokens.
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))

        # Step 5: Add Location Data
        # Now that every token is in its correct grid spot, inject the row/col position embeddings to ALL of them.
        x = x + self.decoder_pos_embed

        # Step 6: Decoder Process
        # The Self-Attention mechanism will heavily look at the 49 visible tokens and figure out what the 147 mask tokens should look like based on context.
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Step 7: Final Linear layer to 'paint' the pixels.
        # Outputs 196 patches, each with 768 raw RGB values.
        x = self.pred(x)  
        return x

# %% [markdown]
# ## Part 5: Full Masked Autoencoder Model
# 
# %% [markdown]
# # Cell 5.1 — Positional Embeddings & MaskedAutoencoder Class
# 

# %%
# ==============================================================================
# PART 5: FULL MASKED AUTOENCODER COMBINATION
# The master class that links the Encoder, Decoder, and Loss Function together.
# ==============================================================================

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Helper utility for positional embeddings.
    Rather than learning position embeddings from scratch, we use fixed Sine/Cosine waves 
    of varying frequencies. This allows the model to easily distinguish horizontal vs vertical positioning,
    and it is mathematically similar to Fourier transforms used in signal processing.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # Create 2D grid coordinates
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    # Mathematical computation of 2D sin-cos sequences
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1) # Combine X and Y coordinates: (196, 768)

    if cls_token: # If we use a CLS token (not required for MAE but good for classification)
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Calculates 1D sine/cosine waves for a specific axis."""
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= (embed_dim / 2.)
    # Frequency scaling mimicking the standard 'Attention is All You Need' paper
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    # Evens = Sine, Odds = Cosine
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)

class MaskedAutoencoder(nn.Module):
    """
    The master pipeline wrapper. 
    Handles taking an image, running it through the entire forward process, AND calculating the MSE loss natively.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                 decoder_embed_dim=384, decoder_depth=12, decoder_num_heads=6,
                 mlp_ratio=4.0, norm_layer=nn.LayerNorm, mask_ratio=0.75):
        super().__init__()
        
        # Instantiate the two massive sub-networks
        self.encoder = ViTEncoder(
            img_size, patch_size, in_channels, encoder_embed_dim,
            encoder_depth, encoder_num_heads, mlp_ratio
        )
        self.decoder = ViTDecoder(
            self.encoder.num_patches, patch_size, encoder_embed_dim,
            decoder_embed_dim, decoder_depth, decoder_num_heads, mlp_ratio
        )
        
        # Save useful configuration parameters
        self.patch_size = patch_size
        self.img_size = img_size

    def forward(self, imgs, mask_ratio=0.75):
        """
        The complete training step computation for one batch of images.
        """
        
        # 1. ENCODER PASS: Generate rich representations for ONLY the 49 visible patches.
        # Outputs latent tokens, the binary mask mapping what was dropped, and the unshuffling `ids_restore` sequence.
        latent, mask, ids_restore = self.encoder(imgs, ids_keep=None)

        # 2. DECODER PASS: Take the 49 visible encodings, inject 147 mask tokens, unshuffle them 
        # using `ids_restore`, and predict all 196 patches of pixel data.
        pred_pixels = self.decoder(latent, ids_restore)

        # 3. COMPUTE LOSS NATIVELY IN THE MODEL
        # We need the Ground Truth to compare against. Calculate the true pixel values matching our patches.
        target_pixels = patchify(imgs, self.patch_size)

        # Normalization trick: As per the MAE paper authors, calculating the MSE on *normalized* pixels per patch
        # improves representation quality significantly.
        # Calculate mean and variance of pixels within EACH PATCH independently.
        mean = target_pixels.mean(dim=-1, keepdim=True)
        var = target_pixels.var(dim=-1, keepdim=True)
        # Normalize the ground truth patch pixels: N(0, 1)
        target_normalized = (target_pixels - mean) / (var + 1e-6).sqrt()

        # Calculate Mean Squared Error (MSE) between Model Predictions and Normalized Ground Truth
        loss = (pred_pixels - target_normalized) ** 2  # Shape: (Batch, 196 patches, 768 pixels)
        loss = loss.mean(dim=-1)  # Average pixel error per patch: Shape (Batch, 196 patches)

        # SUPER IMPORTANT: The loss is ONLY calculated on the masked patches! 
        # If the model perfectly reconstructs the visible patches, we don't care, it was given that data!
        # We multiply the loss vector by the binary mask (where 1=Masked, 0=Visible) to zero out visible errors.
        loss = (loss * mask).sum() / mask.sum()  # Average loss over all masked patches in the batch.

        # Return the aggregated loss, the raw predictions, and the mask
        return loss, pred_pixels, mask

# %% [markdown]
# # Cell 5.2 — Model Instantiation & Parameter Count

# %%
# === Model instantiation & parameter count ===

model = MaskedAutoencoder(
    img_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    in_channels=3,
    encoder_embed_dim=768,
    encoder_depth=12,
    encoder_num_heads=12,
    decoder_embed_dim=384,
    decoder_depth=12,
    decoder_num_heads=6,
    mlp_ratio=4.0,
    mask_ratio=MASK_RATIO,
)

# Count parameters
def count_params(module, name=""):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"{name}: {total:,} total, {trainable:,} trainable")
    return total

encoder_params = sum(p.numel() for name, p in model.named_parameters()
                     if 'decoder' not in name and 'mask_token' not in name)
decoder_params = sum(p.numel() for name, p in model.named_parameters()
                     if 'decoder' in name or 'mask_token' in name)
total_params = sum(p.numel() for p in model.parameters())

print(f"Encoder parameters: {encoder_params:,} (~{encoder_params/1e6:.1f}M)")
print(f"Decoder parameters: {decoder_params:,} (~{decoder_params/1e6:.1f}M)")
print(f"Total parameters:   {total_params:,} (~{total_params/1e6:.1f}M)")

# %% [markdown]
# # Cell 5.3 — Forward Pass Test
# 

# %%
# Quick forward pass test
print("\n=== Forward Pass Test ===")
with torch.no_grad():
    test_imgs = torch.randn(2, 3, 224, 224)
    loss, pred, mask = model(test_imgs)
    print(f"Input:  {test_imgs.shape}")
    print(f"Pred:   {pred.shape}")  # (2, 196, 768)
    print(f"Mask:   {mask.shape}")  # (2, 196)
    print(f"Loss:   {loss.item():.4f}")
    print(f"Masked patches per image: {mask.sum(dim=1).tolist()}")

# %% [markdown]
# ## Part 6: Training Loop
# %% [markdown]
# # Cell 6.1 — Optimizer, Scheduler & Scaler Setup
# 

# %%
# Move model to GPU(s)
if num_gpus > 1:
    model = nn.DataParallel(model)
model = model.to(device)

# Access base model for saving
base_model = model.module if isinstance(model, nn.DataParallel) else model

# Optimizer: AdamW
optimizer = torch.optim.AdamW(model.parameters(),
                               lr=LEARNING_RATE,
                               weight_decay=WEIGHT_DECAY,
                               betas=(0.9, 0.95))

# Scheduler: Cosine Annealing
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# Mixed precision scaler
scaler = GradScaler()

# Gradient clipping value
MAX_GRAD_NORM = 1.0

# %% [markdown]
# # Cell 6.2 — Training & Validation Functions
# 

# %%
# ==============================================================================
# PART 6: TRAINING ENGINE
# This section defines the main logic for updating model weights during training.
# Since MAE is computationally heavy, we use Mixed Precision (FP16) for a 2x speedup.
# ==============================================================================

def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch):
    """
    Executes one complete pass over the entire training dataset.
    Updates the model weights based on the loss.
    """
    model.train()  # Tell PyTorch we are training (enables Dropout gradients, BatchNorm tracking, etc.)
    total_loss = 0.0
    num_batches = 0

    # enumerate() gives us the batch index plus the actual batches of images.
    for batch_idx, images in enumerate(dataloader):
        # Move the raw images to our GPU asynchronously
        images = images.to(device, non_blocking=True)

        # 1. Zero out old gradients from the previous step. If not, they infinitely accumulate!
        optimizer.zero_grad()

        # 2. MIXED PRECISION FORWARD PASS
        # autocast() tells PyTorch to automatically use float16 precision for linear/conv layers
        # where it is mathematically safe to do so. This speeds up T4 GPUs tremendously.
        with autocast():
            loss, pred, mask = model(images)
            loss = loss.mean()  # Critical fix for PyTorch DataParallel on multi-GPU Kaggle targets!
                                # When using 2 GPUs natively, PyTorch returns an array of 2 independent losses.
                                # Averaging them handles this implicitly.

        # 3. BACKWARD PASS & SCALER
        # Since float16 has a very small numeric range, gradients can "underflow" to zero.
        # The GradScaler automatically multiples the loss by a large number, computers derivatives, 
        # and then un-multiplies the results before pushing them into the optimizer.
        scaler.scale(loss).backward()
        
        # 4. GRADIENT CLIPPING
        # Before taking the step, we 'un-scale' the gradients. 
        # If any gradient is explosively high (>1.0), this function forcefully clips it to 1.0. 
        # This prevents the model from diverging uncontrollably ("Exploding Gradients").
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        
        # 5. OPTIMIZER STEP
        # The AdamW optimizer officially adjusts the neural network weights based on the computed gradients.
        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics for reporting
        total_loss += loss.item() # .item() extracts the standard float value off the computation graph
        num_batches += 1

        # Print a progress update every 100 mini-batches
        if (batch_idx + 1) % 100 == 0:
            avg = total_loss / num_batches
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | Avg: {avg:.4f}")

    # Return the average loss calculated over this entire epoch
    return total_loss / num_batches


@torch.no_grad() # Decorator explicitly disabling all gradient tracking. Saves massive amounts of GPU memory during validation!
def validate(model, dataloader, device):
    """
    Evaluates the model on unseen data.
    Does NOT update the model weights. The goal here is solely to test real-world accuracy 
    without "overfitting" or "memorizing" the training set.
    """
    model.eval() # Tell PyTorch we are evaluating (disables Dropout, freezes BatchNorms)
    total_loss = 0.0
    num_batches = 0

    for images in dataloader:
        images = images.to(device, non_blocking=True)
        # We still use autocast during evaluation for inference speedup
        with autocast():
            # In Validation, we calculate how far off our predicted pixels were from the actual original images.
            loss, pred, mask = model(images)
            loss = loss.mean()  # Handle DataParallel vector output
            
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches

# %% [markdown]
# # Cell 6.3 — Main Training Loop

# %%
# === Main Training Loop ===

train_losses = []
val_losses = []
best_val_loss = float('inf')

print("=" * 60)
print("Starting Training")
print("=" * 60)

for epoch in range(NUM_EPOCHS):
    # Train
    train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
    train_losses.append(train_loss)

    # Validate
    val_loss = validate(model, val_loader, device)
    val_losses.append(val_loss)

    # Update LR
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"LR: {current_lr:.6f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': base_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, 'mae_best_model.pth')
        # Also save the pure state_dict as a .pt file
        torch.save(base_model.state_dict(), 'mae_best_weights.pt')
        print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': base_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, f'mae_checkpoint_epoch{epoch+1}.pth')
        # Also save the pure state_dict as a .pt file too
        torch.save(base_model.state_dict(), f'mae_weights_epoch{epoch+1}.pt')
        print(f"  ✓ Checkpoint and weights (.pt) saved at epoch {epoch+1}")

print("\n" + "=" * 60)
print(f"Training Complete! Best Val Loss: {best_val_loss:.4f}")
print("=" * 60)

# %% [markdown]
# ## Part 7: Visualization Module 
# %% [markdown]
# # Cell 7.1 — Loss Plot
# 

# %%
# === Loss Plot ===

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', linewidth=2)
plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Reconstruction Loss (MSE)', fontsize=12)
plt.title('MAE Training: Reconstruction Loss vs Epochs', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# # Cell 7.2 — Reconstruction Visualization (5 Samples)
# 

# %%
# === Reconstruction Visualization ===

@torch.no_grad()
def visualize_reconstruction(model, dataset, device, num_samples=5, mask_ratio=0.75):
    """Show masked input, reconstruction, and original for several images."""
    model.eval()
    base = model.module if isinstance(model, nn.DataParallel) else model
    patch_size = base.patch_size
    img_size = base.img_size

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    col_titles = ['Masked Input (75% removed)', 'Reconstruction', 'Original']

    for i in range(num_samples):
        img = dataset[i].unsqueeze(0).to(device)  # (1, 3, H, W)

        with autocast():
            loss, pred, mask = base(img, mask_ratio=mask_ratio)

        # --- Original image ---
        orig = inv_normalize(img.squeeze(0).cpu())
        orig = orig.permute(1, 2, 0).clamp(0, 1).numpy()

        # --- Masked input visualization ---
        masked_img = create_masked_image(img.squeeze(0).cpu(), mask.squeeze(0).cpu(),
                                          patch_size, img_size)

        # --- Reconstructed image ---
        # Unnormalize predictions (reverse the per-patch normalization)
        target = patchify(img, patch_size)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        pred_unnorm = pred * (var + 1e-6).sqrt() + mean

        # For visualization: use predicted patches for masked, original for visible
        recon_patches = target.clone()
        mask_expanded = mask.squeeze(0).unsqueeze(-1).expand_as(recon_patches.squeeze(0))
        recon_patches[0, mask_expanded[..., 0].bool()] = \
            pred_unnorm[0, mask_expanded[..., 0].bool()]

        recon_img = unpatchify(recon_patches, patch_size, img_size)
        recon_img = inv_normalize(recon_img.squeeze(0).cpu())
        recon_img = recon_img.permute(1, 2, 0).clamp(0, 1).numpy()

        # Full reconstruction from predicted patches
        full_recon = unpatchify(pred_unnorm, patch_size, img_size)
        full_recon = inv_normalize(full_recon.squeeze(0).cpu())
        full_recon = full_recon.permute(1, 2, 0).clamp(0, 1).numpy()

        # Plot
        axes[i, 0].imshow(masked_img)
        axes[i, 1].imshow(full_recon)
        axes[i, 2].imshow(orig)

        for j in range(3):
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=13, fontweight='bold')

    plt.suptitle('MAE Reconstruction Results', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('reconstruction_samples.png', dpi=150, bbox_inches='tight')
    plt.show()


def create_masked_image(img_tensor, mask, patch_size, img_size):
    """Create visualization of masked input image.

    Args:
        img_tensor: (3, H, W) normalized image tensor
        mask: (N,) binary mask (1=masked, 0=visible)
        patch_size: size of each patch
        img_size: full image size
    Returns:
        masked_img: (H, W, 3) numpy array for display
    """
    img = inv_normalize(img_tensor)
    img = img.permute(1, 2, 0).clamp(0, 1).numpy().copy()

    h = w = img_size // patch_size
    for idx in range(len(mask)):
        if mask[idx] == 1:  # masked
            row = idx // w
            col = idx % w
            y1, y2 = row * patch_size, (row + 1) * patch_size
            x1, x2 = col * patch_size, (col + 1) * patch_size
            img[y1:y2, x1:x2] = 0.5  # gray for masked

    return img


# Load best model and visualize
checkpoint = torch.load('mae_best_model.pth', map_location=device)
base_model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

visualize_reconstruction(model, val_dataset, device, num_samples=5)

# %% [markdown]
# ## Part 8: Quantitative Evaluation (PSNR & SSIM)
# %% [markdown]
# # Cell 8.1 — Metric Functions & Evaluation
# 

# %%

def compute_psnr(img1, img2, max_val=1.0):
    """Compute Peak Signal-to-Noise Ratio between two images.

    Args:
        img1, img2: numpy arrays, shape (H, W, 3), range [0, 1]
        max_val: maximum pixel value
    Returns:
        psnr: float
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_ssim(img1, img2, max_val=1.0):
    """Compute Structural Similarity Index (SSIM) between two images.

    Simplified version. For more accurate results, use skimage.

    Args:
        img1, img2: numpy arrays, shape (H, W, 3), range [0, 1]
    Returns:
        ssim: float
    """
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    mu1 = img1.mean(axis=(0, 1))
    mu2 = img2.mean(axis=(0, 1))
    sigma1_sq = ((img1 - mu1) ** 2).mean(axis=(0, 1))
    sigma2_sq = ((img2 - mu2) ** 2).mean(axis=(0, 1))
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(axis=(0, 1))

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim.mean()


@torch.no_grad()
def evaluate_metrics(model, dataset, device, num_samples=50, mask_ratio=0.75):
    """Compute PSNR and SSIM across multiple samples."""
    model.eval()
    base = model.module if isinstance(model, nn.DataParallel) else model
    patch_size = base.patch_size
    img_size = base.img_size

    psnr_scores = []
    ssim_scores = []

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        img = dataset[idx].unsqueeze(0).to(device)

        with autocast():
            loss, pred, mask = base(img, mask_ratio=mask_ratio)

        # Unnormalize predictions
        target = patchify(img, patch_size)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        pred_unnorm = pred * (var + 1e-6).sqrt() + mean

        # Reconstruct images
        orig = inv_normalize(img.squeeze(0).cpu())
        orig = orig.permute(1, 2, 0).clamp(0, 1).numpy()

        recon = unpatchify(pred_unnorm, patch_size, img_size)
        recon = inv_normalize(recon.squeeze(0).cpu())
        recon = recon.permute(1, 2, 0).clamp(0, 1).numpy()

        psnr_scores.append(compute_psnr(orig, recon))
        ssim_scores.append(compute_ssim(orig, recon))

    print(f"\n{'='*50}")
    print(f"Quantitative Evaluation ({num_samples} samples)")
    print(f"{'='*50}")
    print(f"PSNR: {np.mean(psnr_scores):.2f} ± {np.std(psnr_scores):.2f} dB")
    print(f"SSIM: {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
    print(f"{'='*50}")

    return psnr_scores, ssim_scores


# Run evaluation
psnr_scores, ssim_scores = evaluate_metrics(model, val_dataset, device, num_samples=100)

# %% [markdown]
# # Cell 8.2 — Metrics Distribution Plot
# 

# %%
# Plot PSNR and SSIM distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(psnr_scores, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
ax1.axvline(np.mean(psnr_scores), color='red', linestyle='--',
            label=f'Mean: {np.mean(psnr_scores):.2f} dB')
ax1.set_xlabel('PSNR (dB)')
ax1.set_ylabel('Count')
ax1.set_title('PSNR Distribution')
ax1.legend()

ax2.hist(ssim_scores, bins=20, color='coral', edgecolor='white', alpha=0.8)
ax2.axvline(np.mean(ssim_scores), color='red', linestyle='--',
            label=f'Mean: {np.mean(ssim_scores):.4f}')
ax2.set_xlabel('SSIM')
ax2.set_ylabel('Count')
ax2.set_title('SSIM Distribution')
ax2.legend()

plt.suptitle('Reconstruction Quality Metrics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('metrics_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Part 9: Gradio App Deployment
# 
# This cell creates an interactive Gradio app for MAE inference.
# You can also run this as a standalone script: `python gradio_app.py`

# %%
# === Gradio App for MAE Inference ===

import os
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from PIL import Image
from torchvision import transforms

# ---- Paste or import the MAE model classes here (or import from mae_assignment) ----
# If running standalone, you need to copy the model definitions above into this file,
# or use:  from mae_assignment import MaskedAutoencoder, patchify, unpatchify, ...
# For Kaggle notebook usage, this cell runs after all model definitions are available.


# ==================== APP SETUP ====================

# Normalization transforms
normalize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

inv_normalize_fn = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.],
                         std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                         std=[1., 1., 1.]),
])


def load_model(checkpoint_path='mae_best_model.pth'):
    """Load the trained MAE model."""
    model = MaskedAutoencoder(
        img_size=224, patch_size=16, in_channels=3,
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=384, decoder_depth=12, decoder_num_heads=6,
        mlp_ratio=4.0, mask_ratio=0.75,
    )

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from {checkpoint_path}")
    else:
        print(f"⚠ No checkpoint found at {checkpoint_path}. Using random weights.")

    model.eval()
    return model


# Load model globally
mae_model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mae_model = mae_model.to(device)


def create_masked_visualization(img_tensor, mask, patch_size=16, img_size=224):
    """Create masked image for display."""
    img = inv_normalize_fn(img_tensor.cpu())
    img = img.permute(1, 2, 0).clamp(0, 1).numpy().copy()

    h = w = img_size // patch_size
    for idx in range(len(mask)):
        if mask[idx] == 1:
            row = idx // w
            col = idx % w
            y1, y2 = row * patch_size, (row + 1) * patch_size
            x1, x2 = col * patch_size, (col + 1) * patch_size
            img[y1:y2, x1:x2] = 0.5

    return (img * 255).astype(np.uint8)


def process_image(input_image, masking_ratio):
    """Process an uploaded image through the MAE model.

    Args:
        input_image: PIL Image from Gradio upload
        masking_ratio: float, fraction of patches to mask (0.0 to 0.95)
    Returns:
        Tuple of (masked_image, reconstruction, original) as numpy arrays
    """
    if input_image is None:
        return None, None, None

    # Convert to PIL if numpy
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)

    # Transform
    img_tensor = normalize(input_image)  # (3, 224, 224)
    img_batch = img_tensor.unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        loss, pred, mask = mae_model(img_batch, mask_ratio=masking_ratio)

    # Original image
    orig = inv_normalize_fn(img_tensor)
    orig = orig.permute(1, 2, 0).clamp(0, 1).numpy()
    orig_display = (orig * 255).astype(np.uint8)

    # Masked image
    masked_display = create_masked_visualization(
        img_tensor, mask.squeeze(0).cpu(), patch_size=16, img_size=224)

    # Reconstruction
    target = patchify(img_batch, 16)
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)
    pred_unnorm = pred * (var + 1e-6).sqrt() + mean

    recon_img = unpatchify(pred_unnorm, 16, 224)
    recon = inv_normalize_fn(recon_img.squeeze(0).cpu())
    recon = recon.permute(1, 2, 0).clamp(0, 1).numpy()
    recon_display = (recon * 255).astype(np.uint8)

    return masked_display, recon_display, orig_display


# ==================== GRADIO INTERFACE ====================

with gr.Blocks(
    title="Masked Autoencoder (MAE) — Image Reconstruction",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="violet",
    ),
    css="""
    .main-header {
        text-align: center;
        margin-bottom: 20px;
    }
    .output-images img {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
    }
    """
) as demo:

    gr.Markdown("""
    # 🎭 Masked Autoencoder (MAE)
    ### Self-Supervised Image Reconstruction
    
    Upload an image and adjust the masking ratio to see how the MAE reconstructs
    the masked portions. The model was trained on **TinyImageNet** with a **75% masking ratio**.
    
    ---
    """, elem_classes="main-header")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📤 Upload Image",
                type="pil",
                height=300,
            )
            mask_slider = gr.Slider(
                minimum=0.1,
                maximum=0.95,
                value=0.75,
                step=0.05,
                label="🎯 Masking Ratio",
                info="Fraction of patches to mask (default: 0.75 = 75%)"
            )
            run_btn = gr.Button("🚀 Reconstruct", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Row():
                masked_output = gr.Image(label="🔲 Masked Input", height=250)
                recon_output = gr.Image(label="🔧 Reconstruction", height=250)
                orig_output = gr.Image(label="🖼️ Original", height=250)

    gr.Markdown("""
    ---
    **Architecture**: ViT-Base Encoder (~86M params) + ViT-Small Decoder (~22M params)  
    **Training**: AdamW + Cosine LR, MSE loss on masked patches only  
    **GenAI Assignment 02 — Spring 2026**
    """)

    # Event handlers
    run_btn.click(
        fn=process_image,
        inputs=[input_image, mask_slider],
        outputs=[masked_output, recon_output, orig_output],
    )

    # Also trigger on image upload
    input_image.change(
        fn=process_image,
        inputs=[input_image, mask_slider],
        outputs=[masked_output, recon_output, orig_output],
    )

    # Also trigger on slider change (if image is already uploaded)
    mask_slider.release(
        fn=process_image,
        inputs=[input_image, mask_slider],
        outputs=[masked_output, recon_output, orig_output],
    )

# Launch the app
print("\n🚀 Launching Gradio App...")
demo.launch(share=True, server_name="0.0.0.0")

# %% [markdown]
# ## Cell 9 — Summary
# # 
# # | Metric | Value |
# # |--------|-------|
# # | Architecture | MAE (ViT-Base Encoder + ViT-Small Decoder) |
# # | Encoder | 768-dim, 12 layers, 12 heads (~86M params) |
# # | Decoder | 384-dim, 12 layers, 6 heads (~22M params) |
# # | Mask Ratio | 75% |
# # | Loss Function | MSE (masked patches only) |
# # | Optimizer | AdamW |
# # | Scheduler | Cosine Annealing |
# # | Dataset | TinyImageNet (200 classes) |