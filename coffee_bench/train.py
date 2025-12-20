import tensorflow as tf
import os, json
import numpy as np

from .eval import predict_classes, make_report
from .timing import measure_predict_time

def compile_model(model, lr=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def default_callbacks(out_dir, monitor="val_accuracy"):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{out_dir}/best.keras",
            monitor=monitor,
            save_best_only=True,
            mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=5,
            restore_best_weights=True,
            mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=2,
            mode="max",
            verbose=1
        )
    ]

def train_model(model, train_gen, val_gen, epochs=5, callbacks=None, verbose=1):
    return model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks or [],
        verbose=verbose
    )

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def _save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def train_and_log(
    model,
    model_key: str,
    train_gen,
    val_gen,
    out_root: str,
    epochs: int = 5,
    lr: float = 1e-4,
    callbacks=None,
    verbose: int = 1
):
    """
    Train + simpan artefak lengkap per model.
    """
    out_dir = _ensure_dir(os.path.join(out_root, model_key))

    # 1) compile & callbacks
    compile_model(model, lr=lr)  # dari train.py existing :contentReference[oaicite:6]{index=6}
    cbs = callbacks or default_callbacks(out_dir)  # existing :contentReference[oaicite:7]{index=7}

    # 2) fit
    history = train_model(model, train_gen, val_gen, epochs=epochs, callbacks=cbs, verbose=verbose)  # :contentReference[oaicite:8]{index=8}
    _save_json(history.history, os.path.join(out_dir, "history.json"))

    # 3) load best weights (checkpoint sudah simpan best.keras)
    best_path = os.path.join(out_dir, "best.keras")
    if os.path.exists(best_path):
        try:
            import tensorflow as tf
            best_model = tf.keras.models.load_model(best_path)
        except Exception:
            best_model = model
    else:
        best_model = model

    # 4) eval classification report + confusion matrix
    y_true, y_pred, probs = predict_classes(best_model, val_gen)  # :contentReference[oaicite:9]{index=9}
    class_names = list(val_gen.class_indices.keys())
    rep, cm = make_report(y_true, y_pred, class_names)  # :contentReference[oaicite:10]{index=10}

    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(rep)
    np.save(os.path.join(out_dir, "cm.npy"), cm)

    # 5) basic metrics
    val_acc = float(np.mean(y_true == y_pred)) if len(y_true) else None
    metrics = {
        "model_key": model_key,
        "best_path": best_path if os.path.exists(best_path) else None,
        "val_acc": val_acc
    }
    _save_json(metrics, os.path.join(out_dir, "metrics.json"))

    # 6) timing
    try:
        timing = measure_predict_time(best_model, val_gen)  # :contentReference[oaicite:11]{index=11}
    except Exception:
        timing = {"total_sec": None, "n_images": int(getattr(val_gen, "samples", 0)), "sec_per_image": None, "img_per_sec": None}
    _save_json(timing, os.path.join(out_dir, "timing.json"))

    # 7) return row summary
    row = {**metrics, **timing}
    return row


def benchmark_models(
    model_keys,
    build_model_fn,
    make_gens_fn,
    data_dirs: dict,
    out_root: str,
    img_size=(128, 128),
    batch_size=32,
    epochs=5,
    lr=1e-4,
    augment=True,
    verbose=1,
    seed=42
):
    """
    Benchmark multiple backbones dengan pola: generator -> build model -> train_and_log
    """
    _ensure_dir(out_root)
    rows = []

    for k in model_keys:
        train_gen, val_gen, _ = make_gens_fn(
            train_dir=data_dirs["train"],
            val_dir=data_dirs["val"],
            test_dir=data_dirs.get("test"),
            img_size=img_size,
            batch_size=batch_size,
            model_key=k,
            augment=augment,
            seed=seed
        )

        model = build_model_fn(
            model_key=k,
            num_classes=train_gen.num_classes,
            img_size=img_size
        )

        row = train_and_log(
            model=model,
            model_key=k,
            train_gen=train_gen,
            val_gen=val_gen,
            out_root=out_root,
            epochs=epochs,
            lr=lr,
            callbacks=None,
            verbose=verbose
        )
        rows.append(row)

    # save summary.csv
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_root, "summary.csv"), index=False)
    return df
