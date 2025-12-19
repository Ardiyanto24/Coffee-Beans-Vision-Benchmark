import tensorflow as tf
from tensorflow.keras import layers, models

from tensorflow.keras.applications import ResNet50, InceptionV3, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# EfficientNet family (B0-B7 tersedia di tf.keras.applications)
from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
)
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess


def get_preprocess_fn(model_key: str):
    """Return preprocessing function appropriate for each backbone."""
    model_key = model_key.lower()
    if model_key == "resnet50":
        return resnet_preprocess
    if model_key == "inceptionv3":
        return inception_preprocess
    if model_key == "mobilenetv2":
        return mobilenet_preprocess
    if model_key.startswith("efficientnet"):
        return effnet_preprocess
    if model_key == "cnn_basic":
        # For simple CNN, use [0..255] -> [0..1]
        return lambda x: x / 255.0
    raise ValueError(f"Unknown model_key: {model_key}")


def build_cnn_basic(num_classes: int, img_size=(128, 128)):
    model = models.Sequential([
        layers.Input(shape=(*img_size, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model


def _build_transfer_head(base_model, num_classes: int, dense_units=256, dropout=0.4):
    base_model.trainable = False
    inputs = layers.Input(shape=base_model.input_shape[1:])
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)


def build_model(model_key: str,
                num_classes: int,
                img_size=(128, 128),
                dense_units=256,
                dropout=0.4,
                unfreeze_last=0):
    """
    model_key:
      - cnn_basic
      - resnet50
      - mobilenetv2
      - inceptionv3
      - efficientnetb0/b1/b2/b3
    """
    k = model_key.lower()

    if k == "cnn_basic":
        return build_cnn_basic(num_classes=num_classes, img_size=img_size)

    input_shape = (*img_size, 3)

    if k == "resnet50":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
        model = _build_transfer_head(base, num_classes, dense_units, dropout)
    elif k == "mobilenetv2":
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
        model = _build_transfer_head(base, num_classes, dense_units, dropout)
    elif k == "inceptionv3":
        base = InceptionV3(weights="imagenet", include_top=False, input_shape=input_shape)
        model = _build_transfer_head(base, num_classes, dense_units, dropout)
    elif k == "efficientnetb0":
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
        model = _build_transfer_head(base, num_classes, dense_units, dropout)
    elif k == "efficientnetb1":
        base = EfficientNetB1(weights="imagenet", include_top=False, input_shape=input_shape)
        model = _build_transfer_head(base, num_classes, dense_units, dropout)
    elif k == "efficientnetb2":
        base = EfficientNetB2(weights="imagenet", include_top=False, input_shape=input_shape)
        model = _build_transfer_head(base, num_classes, dense_units, dropout)
    elif k == "efficientnetb3":
        base = EfficientNetB3(weights="imagenet", include_top=False, input_shape=input_shape)
        model = _build_transfer_head(base, num_classes, dense_units, dropout)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    # Optional fine-tuning: unfreeze last N layers
    if unfreeze_last and k != "cnn_basic":
        # unfreeze last N layers of base model
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break
        # di model hasil _build_transfer_head, base model adalah layer ke-1 (umumnya)
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break
        if base_model is None:

    return model
