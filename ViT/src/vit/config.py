from dataclasses import dataclass


@dataclass
class ViTConfig:
    # --- image / patching ---
    image_size: int = 32
    patch_size: int = 4          # 32 / 4 = 8 -> 8 * 8 = 64 patches
    in_channels: int = 3

    # --- transformer ---
    embed_dim: int = 128
    depth: int = 6               # number of encoder blocks
    num_heads: int = 8           # head_dim = 128 / 8 = 16
    mlp_ratio: float = 4.0       # mlp hidden = embed_dim * mlp_ratio (same 4x as your mini-transformer)
    dropout: float = 0.1
    attn_dropout: float = 0.0

    # --- task head ---
    num_classes: int = 10

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    label_smoothing: float = 0.1
    num_workers: int = 0         # keep 0 on Windows (see the multiprocessing error from Phase 2)
    seed: int = 42
    data_dir: str = "./data"
    out_dir: str = "./outputs"
