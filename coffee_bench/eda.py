import os, random, shutil, hashlib
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image

# Optional heavy deps (buat blur via cv2). Kalau gak ada, fungsi blur bisa diskip.
try:
    import cv2
except Exception:
    cv2 = None

import matplotlib.pyplot as plt


# ---------- BASIC SCAN ----------
def scan_classes(train_dir: str):
    class_names = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    return class_names


def count_images_per_class(train_dir: str, class_names=None):
    if class_names is None:
        class_names = scan_classes(train_dir)

    counts = {}
    for cls in class_names:
        counts[cls] = len(glob(os.path.join(train_dir, cls, "*")))
    return counts


def build_df_from_folders(train_dir: str, class_names=None):
    if class_names is None:
        class_names = scan_classes(train_dir)

    rows = []
    for cls in class_names:
        paths = glob(os.path.join(train_dir, cls, "*"))
        rows.extend([(p, cls) for p in paths])

    df = pd.DataFrame(rows, columns=["path", "label"])
    return df


# ---------- PLOTS (HRD-friendly) ----------
def plot_samples_grid(train_dir: str, class_names=None, n_samples=3, seed=42):
    random.seed(seed)
    if class_names is None:
        class_names = scan_classes(train_dir)

    fig, axes = plt.subplots(
        nrows=len(class_names),
        ncols=n_samples,
        figsize=(n_samples * 3, len(class_names) * 3)
    )

    if len(class_names) == 1:
        axes = [axes]

    for row_idx, cls in enumerate(class_names):
        class_dir = os.path.join(train_dir, cls)
        img_paths = glob(os.path.join(class_dir, "*"))
        chosen = img_paths if len(img_paths) <= n_samples else random.sample(img_paths, n_samples)

        for col_idx in range(n_samples):
            ax = axes[row_idx][col_idx] if len(class_names) > 1 else axes[col_idx]
            if col_idx < len(chosen):
                img = Image.open(chosen[col_idx]).convert("RGB")
                ax.imshow(img)
                ax.set_title(cls)
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_label_distribution(df: pd.DataFrame, title="Distribusi Jumlah Gambar per Kelas"):
    counts = df["label"].value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    plt.bar(counts.index, counts.values)
    plt.title(title)
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah Gambar")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


# ---------- DUPLICATE CHECK ----------
def file_hash(path: str):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def add_hash_and_find_duplicates(df: pd.DataFrame):
    df = df.copy()
    df["hash"] = df["path"].apply(file_hash)
    dup_df = df[df.duplicated("hash", keep=False)].sort_values("hash")
    dup_groups = dup_df.groupby("hash")["path"].apply(list)
    return df, dup_df, dup_groups


# ---------- IMAGE STATS ----------
def _get_image_info(path: str):
    img = Image.open(path)
    w, h = img.size
    return w, h, (w / h if h else np.nan)

def add_image_stats(df: pd.DataFrame):
    df = df.copy()
    info = df["path"].apply(_get_image_info)
    df["width"] = info.apply(lambda x: x[0])
    df["height"] = info.apply(lambda x: x[1])
    df["aspect_ratio"] = info.apply(lambda x: x[2])
    return df


def plot_image_stats(df: pd.DataFrame):
    plt.figure(figsize=(8, 4))
    plt.hist(df["width"], bins=30)
    plt.title("Distribusi Lebar Gambar (px)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(df["height"], bins=30)
    plt.title("Distribusi Tinggi Gambar (px)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(df["aspect_ratio"], bins=40)
    plt.axvline(1.0, linestyle="--")
    plt.title("Distribusi Aspect Ratio (W/H)")
    plt.tight_layout()
    plt.show()


# ---------- QUALITY HEURISTICS ----------
def detect_blur(path: str):
    if cv2 is None:
        return np.nan
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.nan
    return cv2.Laplacian(img, cv2.CV_64F).var()

def detect_brightness(path: str):
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return float(arr.mean())

def detect_contrast(path: str):
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return float(arr.std())

def estimate_noise(path: str):
    img = Image.open(path).convert("L")
    arr = np.array(img).astype(np.float32)
    return float(np.median(np.abs(arr - np.median(arr))))

def add_quality_scores(df: pd.DataFrame, use_noise=True):
    df = df.copy()
    df["blur_score"] = df["path"].apply(detect_blur)
    df["brightness"] = df["path"].apply(detect_brightness)
    df["contrast"] = df["path"].apply(detect_contrast)
    if use_noise:
        df["noise"] = df["path"].apply(estimate_noise)
    return df

def add_quality_flags(df: pd.DataFrame,
                      blur_th=50,
                      bright_th=70,
                      contrast_th=30):
    df = df.copy()
    df["bad_blur"] = df["blur_score"] < blur_th
    df["bad_dark"] = df["brightness"] < bright_th
    df["bad_contrast"] = df["contrast"] < contrast_th
    df["bad_quality"] = df["bad_blur"] | df["bad_dark"] | df["bad_contrast"]
    return df

def show_bad_samples(df: pd.DataFrame, n=8, seed=42):
    bad = df[df.get("bad_quality", False)].copy()
    if len(bad) == 0:
        print("Tidak ada bad_quality yang terdeteksi.")
        return
    bad = bad.sample(min(n, len(bad)), random_state=seed)

    plt.figure(figsize=(12, 6))
    for i, (_, row) in enumerate(bad.iterrows(), start=1):
        img = Image.open(row["path"]).convert("RGB")
        plt.subplot(2, 4, i)
        plt.imshow(img)
        plt.title(f"{row['label']}\nblur={row.get('blur_score', np.nan):.1f}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# ---------- SPLIT & MATERIALIZE FOLDERS ----------
def make_stratified_split(df: pd.DataFrame, test_size=0.2, seed=42):
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def materialize_split_folders(train_df: pd.DataFrame,
                              val_df: pd.DataFrame,
                              output_base: str,
                              copy_test: bool = False,
                              test_dir: str = None):
    train_out = os.path.join(output_base, "train")
    val_out = os.path.join(output_base, "val")
    test_out = os.path.join(output_base, "test")

    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)
    if copy_test:
        os.makedirs(test_out, exist_ok=True)

    def _copy_split(split_df, base_out):
        for _, row in split_df.iterrows():
            label = row["label"]
            src_path = row["path"]
            dst_dir = os.path.join(base_out, label)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, os.path.basename(src_path))
            if not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)

    _copy_split(train_df, train_out)
    _copy_split(val_df, val_out)

    if copy_test:
        if test_dir is None:
            raise ValueError("test_dir wajib diisi kalau copy_test=True")
        shutil.copytree(test_dir, test_out, dirs_exist_ok=True)

    return {"train": train_out, "val": val_out, "test": test_out if copy_test else None}

# eda.py (tambahan)
def run_full_eda(train_dir: str, n_samples=3, seed=42, show_bad_n=8):
    class_names = scan_classes(train_dir)
    print("Classes:", class_names)

    df = build_df_from_folders(train_dir, class_names)
    print("Total images:", len(df))

    # 1) sample grid
    plot_samples_grid(train_dir, class_names=class_names, n_samples=n_samples, seed=seed)

    # 2) label distribution
    plot_label_distribution(df)

    # 3) duplicates
    df_h, dup_df, dup_groups = add_hash_and_find_duplicates(df)
    print("Duplicate groups:", len(dup_groups))
    if len(dup_groups) > 0:
        # tampilkan 1 grup contoh
        first_hash = list(dup_groups.index)[0]
        print("Example dup hash:", first_hash)
        for p in dup_groups[first_hash][:5]:
            print(" -", p)

    # 4) image stats
    df_stats = add_image_stats(df)
    plot_image_stats(df_stats)

    # 5) quality heuristics
    df_q = add_quality_scores(df_stats, use_noise=True)
    df_q = add_quality_flags(df_q)
    print("Bad quality count:", int(df_q["bad_quality"].sum()))
    show_bad_samples(df_q, n=show_bad_n, seed=seed)

    return {
        "class_names": class_names,
        "df": df,
        "df_stats": df_stats,
        "df_quality": df_q,
        "dup_groups": dup_groups
    }
