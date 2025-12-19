import time
import numpy as np

def measure_predict_time(model, gen, warmup_batches=1):
    """
    Mengukur total waktu predict pada seluruh generator (val/test) dan waktu per image.
    Tips fairness:
    - Warmup dulu supaya graph/trace tidak ngotorin timing.
    - Pastikan shuffle=False pada gen.
    """
    # Warmup
    for _ in range(warmup_batches):
        x, _ = next(gen)
        _ = model.predict(x, verbose=0)

    # Reset iterator generator supaya predict mulai dari awal
    gen.reset()

    start = time.perf_counter()
    _ = model.predict(gen, verbose=0)
    end = time.perf_counter()

    total_sec = end - start
    n_images = gen.samples
    sec_per_image = total_sec / max(n_images, 1)

    return {
        "total_sec": float(total_sec),
        "n_images": int(n_images),
        "sec_per_image": float(sec_per_image),
        "img_per_sec": float(1.0 / sec_per_image) if sec_per_image > 0 else None
    }
