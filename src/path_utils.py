import numpy as np


def get_val_loss_from_ckpt_path(ckpt_path):
    try:
        val_loss = float(ckpt_path.stem.split("=")[-1])
    except Exception:
        val_loss = np.inf
    return val_loss


def get_best_ckpt_path(rootdir, version: int = -1, is_last: bool = False):
    ckpt_path = rootdir / "lightning_logs"

    best_ckpt_path = None
    best_val_loss = np.inf

    for version_dir in ckpt_path.iterdir():
        if version != -1 and version != int(version_dir.stem.split("_")[-1]):
            continue

        checkpoints_dir = version_dir / "checkpoints"
        if not checkpoints_dir.is_dir():
            continue

        if is_last:
            best_ckpt_path = checkpoints_dir / "last.ckpt"
        else:
            for ckpt_file in checkpoints_dir.iterdir():
                val_loss = get_val_loss_from_ckpt_path(ckpt_file)
                if (best_ckpt_path is None) or (val_loss < best_val_loss):
                    best_ckpt_path = ckpt_file
                    best_val_loss = val_loss

    return best_ckpt_path
