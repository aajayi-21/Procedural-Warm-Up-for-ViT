"""Typed configuration schema for warm-up and downstream stages.

Configs are plain ``@dataclass`` trees with sensible defaults (matching the paper's
ViT-T/16 recipe). A YAML file supplies only the fields it wants to override; the loader
recursively merges it onto the default tree. This mirrors the reference repo's
``cfg.py`` but unifies the warm-up and downstream stages into one module.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Optional

import yaml


# --------------------------------------------------------------------------------------
# Warm-up (procedural pretraining) configuration
# --------------------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """ViT backbone. ``embed_dim`` must match the timm model (ViT-T = 192)."""

    name: str = "vit_tiny_patch16_224"
    embed_dim: int = 192
    num_classes: int = 0  # warm-up has no classifier head; the MLM head is separate
    drop_path_rate: float = 0.0


@dataclass
class GridConfig:
    """Token grid. ``N = H * W`` must equal the ViT's visual-token count (14*14=196)."""

    H: int = 14
    W: int = 14


@dataclass
class VocabConfig:
    """Token vocabulary. Ids 0/1 are reserved for PAD/MASK across all sources."""

    K: int = 130
    PAD_ID: int = 0
    MASK_ID: int = 1


@dataclass
class DyckConfig:
    """k-Dyck grammar generator (reference parity baseline)."""

    k_open: int = 64
    k_close: int = 64
    open_prob: float = 0.6
    min_pairs: int = 1


@dataclass
class CATokenizeConfig:
    """How cellular-automaton cells map to vocabulary ids.

    ``binary`` uses one token per cell state (faithful to the edge-of-chaos binary I/O).
    ``block`` coarse-grains ``block_size`` consecutive cells into a single 2**block_size
    symbol, exercising a larger vocabulary (the CA analog of the k-Dyck vocab sweep).
    """

    mode: str = "binary"  # "binary" | "block"
    block_size: int = 7


@dataclass
class CAConfig:
    """Cellular-automaton data source."""

    rule: int = 110  # ECA rule number (0-255); B/S notation handled separately for GoL
    dim: int = 1  # 1 = elementary (1D) CA spacetime; 2 = Game of Life (Stage-4 scaffold)
    sim_width: int = 64  # simulated torus width (wider than the window => real light cones)
    burn_in: int = 32  # initial rows discarded to escape transients
    boundary: str = "periodic"  # "periodic" | "zero"
    init_density: float = 0.5  # Bernoulli probability for the random initial row
    tokenize: CATokenizeConfig = field(default_factory=CATokenizeConfig)


@dataclass
class MaskingConfig:
    """Masked-token objective. Shared ``mask_ratio`` across sources.

    ``mode`` is interpreted by the CA masking strategy (``random`` | ``forward`` |
    ``lightcone``); the Dyck source ignores it (it always masks closing brackets).
    """

    mode: str = "random"
    mask_ratio: float = 0.5
    forward_rows: int = 6  # for CA ``forward`` mode: number of trailing (future) rows masked


@dataclass
class DataConfig:
    """Selects the procedural data source from the registry."""

    source: str = "ca"  # registry key: "ca" | "dyck" | "gol"


@dataclass
class DatasetConfig:
    n_samples: int = 100_000  # virtual epoch length; the trainer cycles the loader by steps
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TrainingConfig:
    steps: int = 15_000
    device: str = "cuda"  # falls back to cpu automatically if cuda is unavailable


@dataclass
class OptimizerConfig:
    lr: float = 2e-3
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    enabled: bool = True
    warmup_steps: int = 1_000
    warmup_start_lr: float = 1e-6
    min_lr: float = 1e-5


@dataclass
class CheckpointConfig:
    out_dir: str = "checkpoints"
    save_steps: list = field(default_factory=lambda: [15_000])


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "procedural-warmup-ca"
    entity: Optional[str] = None


@dataclass
class LoggingConfig:
    print_freq: int = 50


@dataclass
class StageConfig:
    """A single curriculum stage (Stage-2 scaffold; see ``warmup/curriculum.py``).

    ``overrides`` is a nested dict merged onto the base config for that stage, so a
    curriculum can switch data source and any hyperparameters per round without code
    changes.
    """

    source: str = "ca"
    steps: int = 7_500
    overrides: dict = field(default_factory=dict)


@dataclass
class RootConfig:
    """Top-level warm-up configuration."""

    seed: int = 42
    run_name: str = "ca-rule110"
    results_dir: str = "results"

    model: ModelConfig = field(default_factory=ModelConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    vocab: VocabConfig = field(default_factory=VocabConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dyck: DyckConfig = field(default_factory=DyckConfig)
    ca: CAConfig = field(default_factory=CAConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Stage-2 curriculum scaffold: an empty list means single-stage warm-up.
    curriculum: list = field(default_factory=list)

    @property
    def N(self) -> int:
        """Number of tokens per sample (= ViT visual-token count)."""
        return self.grid.H * self.grid.W


# --------------------------------------------------------------------------------------
# Downstream (image classification) configuration
# --------------------------------------------------------------------------------------


@dataclass
class DownstreamModelConfig:
    name: str = "vit_tiny_patch16_224"
    drop_path_rate: float = 0.1


@dataclass
class DownstreamDataConfig:
    dataset: str = "CIFAR100"  # "CIFAR10" | "CIFAR100"
    data_root: str = "data"
    input_size: int = 224  # resize so the 14x14 patch grid matches the warm-up geometry
    num_workers: int = 8


@dataclass
class AugConfig:
    color_jitter: float = 0.3
    auto_augment: str = "rand-m9-mstd0.5-inc1"
    reprob: float = 0.25  # random-erase probability
    mixup: float = 0.8
    cutmix: float = 1.0
    smoothing: float = 0.1


@dataclass
class DownstreamTrainConfig:
    epochs: int = 300
    warmup_epochs: int = 50
    batch_size: int = 512
    lr: float = 2e-3
    weight_decay: float = 0.05
    min_lr: float = 1e-5
    warmup_lr: float = 1e-6
    clip_grad: float = 1.0
    use_amp: bool = True
    device: str = "cuda"


@dataclass
class DownstreamConfig:
    """Top-level downstream configuration."""

    seed: int = 42
    run_name: str = "cifar100-rule110"
    results_dir: str = "results"
    # Path to a stripped warm-up checkpoint (from warmup/process.py); None = random init.
    init_checkpoint: Optional[str] = None

    model: DownstreamModelConfig = field(default_factory=DownstreamModelConfig)
    data: DownstreamDataConfig = field(default_factory=DownstreamDataConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    train: DownstreamTrainConfig = field(default_factory=DownstreamTrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


# --------------------------------------------------------------------------------------
# Loading / merging
# --------------------------------------------------------------------------------------


def _merge(node: Any, raw: dict) -> None:
    """Recursively override dataclass ``node`` in place with values from ``raw``."""
    for key, value in raw.items():
        if not hasattr(node, key):
            raise KeyError(f"Unknown config key '{key}' for {type(node).__name__}")
        current = getattr(node, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(node, key, tuple(value))
        else:
            setattr(node, key, value)


def _load(cls, path: Optional[str]):
    cfg = cls()
    if path is not None:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        _merge(cfg, raw)
    return cfg


def load_config(path: Optional[str] = None) -> RootConfig:
    """Load a warm-up config, applying ``path``'s overrides onto the defaults."""
    return _load(RootConfig, path)


def load_downstream_config(path: Optional[str] = None) -> DownstreamConfig:
    """Load a downstream (image-training) config."""
    return _load(DownstreamConfig, path)


def config_to_dict(cfg: Any) -> dict:
    """Recursively convert a dataclass config to a plain dict (for YAML/JSON dumps)."""
    if is_dataclass(cfg):
        return {f.name: config_to_dict(getattr(cfg, f.name)) for f in fields(cfg)}
    if isinstance(cfg, (list, tuple)):
        return [config_to_dict(v) for v in cfg]
    return cfg
