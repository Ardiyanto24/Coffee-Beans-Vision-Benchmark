import tensorflow as tf

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
