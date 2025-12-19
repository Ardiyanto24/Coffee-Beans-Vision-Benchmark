from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .models import get_preprocess_fn


def make_gens(train_dir,
              val_dir,
              test_dir=None,
              img_size=(128, 128),
              batch_size=32,
              model_key="resnet50",
              augment=True,
              seed=42):
    """
    Membuat train/val/test generator dengan preprocess yang sesuai backbone model.

    train_dir: folder train yang berisi subfolder kelas
    val_dir  : folder val yang berisi subfolder kelas
    test_dir : (opsional) folder test (biasanya tanpa subfolder kelas)
    """

    preprocess_fn = get_preprocess_fn(model_key)

    # Train datagen
    if augment:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_fn,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )
    else:
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    # Val datagen (no aug)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=seed
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    test_gen = None
    if test_dir is not None:
        # Banyak dataset test Kaggle tidak punya subfolder kelas
        # Jadi kita pakai flow_from_directory dengan class_mode=None
        test_gen = val_datagen.flow_from_directory(
            test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode=None,
            shuffle=False
        )

    return train_gen, val_gen, test_gen
