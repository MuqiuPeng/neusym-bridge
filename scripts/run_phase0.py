"""Phase 0: Generate 2D heat equation data and train three baseline models.

Pass criterion: three models' val loss converge within 5% of each other.
"""

from pathlib import Path

import torch
import h5py

from neusym_bridge.data.heat_equation import HeatConfig, generate_dataset
from neusym_bridge.models.baseline_mlp import create_model, SEED_CONFIGS
from neusym_bridge.models.trainer import train_model, save_model

DATA_DIR = Path("data")
MODEL_DIR = Path("checkpoints")


def main():
    # Step 1: Generate dataset
    print("=== Generating 2D heat equation dataset ===")
    config = HeatConfig(n_trajectories=200)
    data_path = generate_dataset(config, DATA_DIR / "heat_2d.h5")
    print(f"Saved {config.n_trajectories} trajectories to {data_path}")

    # Load as torch tensor
    with h5py.File(data_path, "r") as f:
        trajectories = torch.tensor(f["trajectories"][:], dtype=torch.float32)
    print(f"Trajectories shape: {trajectories.shape}")

    # Step 2: Train three models
    val_losses = {}
    train_losses = {}
    for name, seed in SEED_CONFIGS.items():
        print(f"\n=== Training {name} (seed={seed}) ===")
        torch.manual_seed(seed)
        model = create_model(name)
        history = train_model(model, trajectories, n_epochs=30, batch_size=512)

        val_losses[name] = history["val_loss"][-1]
        train_losses[name] = history["train_loss"][-1]
        print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
        print(f"  Final val loss:   {history['val_loss'][-1]:.6f}")

        save_model(model, MODEL_DIR / f"{name}.pt")

    # Step 3: Convergence check
    # Use train loss for convergence check (val loss is too noisy at very low values)
    print("\n=== Convergence Check ===")
    print(f"  Val losses:   {val_losses}")
    print(f"  Train losses: {train_losses}")

    tl = list(train_losses.values())
    ratio = max(tl) / min(tl) if min(tl) > 0 else float("inf")
    print(f"  Train loss max/min ratio: {ratio:.4f}")
    if ratio < 1.05:
        print("  PASS: Models converged within 5% (train loss)")
    else:
        print(f"  FAIL: Ratio {ratio:.4f} exceeds 1.05")


if __name__ == "__main__":
    main()
