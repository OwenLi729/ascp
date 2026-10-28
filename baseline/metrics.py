"""Common evaluation metrics for nested classification prediction sets."""
import numpy as np


def evaluate(prediction_masks, labels):
    masks = np.asarray(prediction_masks, dtype=bool)
    labels = np.asarray(labels)
    if (masks.ndim != 3 or min(masks.shape) < 1 or labels.shape != (len(masks),)
            or not np.issubdtype(labels.dtype, np.integer)
            or ((labels < 0) | (labels >= masks.shape[2])).any()):
        raise ValueError("provide (n, stages, classes) masks and integer labels")
    if (masks[:, 1:] & ~masks[:, :-1]).any():
        raise ValueError("prediction sets must be nested")
    stages = masks.shape[1]
    included = masks[np.arange(len(labels))[:, None], np.arange(stages), labels[:, None]]
    sizes = masks.sum(axis=2)
    singleton = sizes == 1
    stopping = np.where(singleton.any(axis=1), singleton.argmax(axis=1) + 1, stages + 1)
    return {
        "amr": float((~included).any(axis=1).mean()),
        "cumulative_miscoverage": np.maximum.accumulate(~included, axis=1).mean(axis=0).tolist(),
        "mean_set_size": sizes.mean(axis=0).tolist(),
        "empty_set_rate": (sizes == 0).mean(axis=0).tolist(),
        "mean_stopping_stage": float(stopping.mean()),
        "unresolved_rate": float((stopping == stages + 1).mean()),
    }
