import tensorflow as tf
from tensorflow.keras import layers, models

from tensorflow.keras.applications import ResNet50, InceptionV3, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
)
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess


def get_preprocess_fn(model_key: str):
    """
    Return preprocessing function appropriate for each backbone.
    model_key valid:
      - resnet50
      - inceptionv3
      - mobilenetv2
      - efficientnetb0/b1/b2/b3 (atau string yg diawali 'efficientnet')
      - cnn_basic
    """
    k = (model_key or "").lower()

    if k == "resnet50":
        return resnet_preprocess
    if k == "inceptionv3":
        return inception_preprocess
    if k == "mobilenetv2":
        return mobilenet_preprocess
    if k.startswith("efficientnet"):
        return effnet_preprocess
    if k == "cnn_basic":
        # Simple CNN: normalize [0..255] -> [0..1]
        return lambda x: x / 255.0

    raise ValueError(f"Unknown model_key for preprocess: {model_key}")


def build_cnn_basic(num_classes: int, img_size=(128, 128)):
    """Basic CNN baseline."""
    return models.Sequential([
        layers.Input(shape=(*img_size, 3)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])


def _build_transfer_head(base_model, num_classes: int, dense_units=256, dropout=0.4):
    """
    Build a classification head on top of a frozen base model.
    """
    base_model.trainable = False

    inputs = layers.Input(shape=base_model.input_shape[1:])
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)


def _get_base_model(model_key: str, input_shape):
    """
    Helper: return the backbone model instance based on model_key.
    """
    k = model_key.lower()

    if k == "resnet50":
        return ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    if k == "mobilenetv2":
        return MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    if k == "inceptionv3":
        return InceptionV3(weights="imagenet", include_top=False, input_shape=input_shape)
    if k == "efficientnetb0":
        return EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    if k == "efficientnetb1":
        return EfficientNetB1(weights="imagenet", include_top=False, input_shape=input_shape)
    if k == "efficientnetb2":
        return EfficientNetB2(weights="imagenet", include_top=False, input_shape=input_shape)
    if k == "efficientnetb3":
        return EfficientNetB3(weights="imagenet", include_top=False, input_shape=input_shape)

    raise ValueError(f"Unknown transfer model_key: {model_key}")


def build_model(
    model_key: str,
    num_classes: int,
    img_size=(128, 128),
    dense_units=256,
    dropout=0.4,
    unfreeze_last=0
):
    """
    Build model for benchmarking.

    model_key:
      - cnn_basic
      - resnet50
      - mobilenetv2
      - inceptionv3
      - efficientnetb0/b1/b2/b3
    """
    k = (model_key or "").lower()

    # 1) Basic CNN
    if k == "cnn_basic":
        return build_cnn_basic(num_classes=num_classes, img_size=img_size)

    # 2) Transfer learning models
    input_shape = (*img_size, 3)
    base = _get_base_model(k, input_shape=input_shape)
    model = _build_transfer_head(base, num_classes, dense_units, dropout)

    # 3) Optional fine-tuning (unfreeze last N layers of base model)
    if unfreeze_last and unfreeze_last > 0:
        # base model adalah layer ke-1 pada model hasil _build_transfer_head:
        # [InputLayer, base_model, GlobalAveragePooling2D, Dense, Dropout, Dense]
        base_model = model.layers[1]
        base_model.trainable = True

        # freeze semua kecuali last N
        if unfreeze_last < len(base_model.layers):
            for layer in base_model.layers[:-unfreeze_last]:
                layer.trainable = False

    return model
