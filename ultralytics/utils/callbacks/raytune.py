# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import SETTINGS

try:
    assert SETTINGS["raytune"] is True  # verify integration is enabled
    import ray
    from ray import tune
    from ray.air import session

except (ImportError, AssertionError):
    tune = None


def _has_session():
    """Return True when training is running inside an active Ray session."""
    getters = []

    train_internal = getattr(getattr(ray, "train", None), "_internal", None)
    train_session = getattr(train_internal, "session", None) if train_internal else None
    for name in ("get_session", "_get_session"):
        getter = getattr(train_session, name, None) if train_session else None
        if callable(getter):
            getters.append(getter)

    air_getter = getattr(session, "get_session", None)
    if callable(air_getter):
        getters.append(air_getter)

    for getter in getters:
        try:
            if getter():
                return True
        except Exception:
            continue
    return False


def on_fit_epoch_end(trainer):
    """Sends training metrics to Ray Tune at end of each epoch."""
    if _has_session():
        metrics = trainer.metrics
        metrics["epoch"] = trainer.epoch
        session.report(metrics)


callbacks = (
    {
        "on_fit_epoch_end": on_fit_epoch_end,
    }
    if tune
    else {}
)
