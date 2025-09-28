# make_predictions_no2.py — standardize target (stable, no overflow, no flat fields)
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from datetime import date, timedelta

CSV_HISTORY = "pm25_standard_2022_2024.csv" #vale edw auto p thes
OUTPUT_CSV  = "pm25_prediction.csv"
RANDOM_SEED = 42

# Prediction window (inclusive)
PRED_YEAR  = 2025
PRED_START = (PRED_YEAR, 10, 1)
PRED_END   = (PRED_YEAR, 10, 7)

# Grid bbox (auto-fitted to history by default)
AUTO_FIT_BBOX = True
LAT_MIN, LAT_MAX = 34.0, 42.0
LON_MIN, LON_MAX = 18.0, 28.0
STEP_DEG = 0.2

# Training
EPOCHS = 200
BATCH  = 64
LR     = 1e-4
OUTLIER_Q = 0.999  # drop top 0.1% of values to stabilize

# Output clamping (optional). Leave both as None to disable.
CLIP_MIN = None
CLIP_MAX_MULT = None  # e.g., 1.25 caps at 1.25 × 99.9th percentile of training values

tf.keras.backend.clear_session()
tf.random.set_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

def daterange(d0, d1):
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)

def build_model(input_dim, normalizer):
    x = keras.Input(shape=(input_dim,), name="features")
    h = normalizer(x)
    h = keras.layers.Dense(128, activation="relu")(h)
    h = keras.layers.Dense(64,  activation="relu")(h)
    y = keras.layers.Dense(1, activation=None)(h)  # linear; predicts standardized target
    model = keras.Model(x, y)
    opt = keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0)  # gradient clip for safety
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model

def main():
    # ----- Load & clean -----
    df = pd.read_csv(CSV_HISTORY)

    for c in ["time","latitude","longitude","value"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in ["latitude","longitude","value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time","latitude","longitude","value"])
    df = df[df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)]

    # Trim extreme outliers in raw space (helps stability)
    v_hi = df["value"].quantile(OUTLIER_Q)
    if np.isfinite(v_hi):
        df = df[df["value"] <= v_hi]

    # ----- Features -----
    df["year"]        = df["time"].dt.year.astype(int)
    df["day_of_week"] = df["time"].dt.dayofweek.astype(int)
    feature_cols = ["latitude","longitude","year","day_of_week"]

    # ----- Standardize target (z-score) -----
    y_raw  = df["value"].astype(np.float32).to_numpy()
    y_mean = float(np.mean(y_raw))
    y_std  = float(np.std(y_raw))
    if y_std < 1e-8:  # avoid divide-by-zero on very flat datasets
        y_std = 1.0
    df["_target"] = (y_raw - y_mean) / y_std

    # ----- Split (year-aware; fallback to random) -----
    years = sorted(df["year"].unique())
    if len(years) >= 2:
        last = years[-1]
        train_df = df[df["year"] < last].copy()
        val_df   = df[df["year"] == last].copy()
        if train_df.empty:
            train_df = df.sample(frac=0.8, random_state=RANDOM_SEED)
            val_df   = df.drop(train_df.index).copy()
    else:
        train_df = df.sample(frac=0.8, random_state=RANDOM_SEED)
        val_df   = df.drop(train_df.index).copy()

    X_train = train_df[feature_cols].to_numpy(np.float32)
    y_train = train_df["_target"].to_numpy(np.float32)
    X_val   = val_df[feature_cols].to_numpy(np.float32)
    y_val   = val_df["_target"].to_numpy(np.float32)

    # ----- Normalizer + Model -----
    norm = keras.layers.Normalization(axis=-1); norm.adapt(X_train)
    model = build_model(X_train.shape[1], norm)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=300, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(X_val) else None,
        epochs=EPOCHS, batch_size=BATCH, shuffle=True, verbose=1, callbacks=[es]
    )

    # ----- BBox for predictions -----
    if AUTO_FIT_BBOX:
        lat_min = float(df["latitude"].min()); lat_max = float(df["latitude"].max())
        lon_min = float(df["longitude"].min()); lon_max = float(df["longitude"].max())
        lat_pad = 0.02 * max(1e-6, lat_max - lat_min)
        lon_pad = 0.02 * max(1e-6, lon_max - lon_min)
        lat_min -= lat_pad; lat_max += lat_pad
        lon_min -= lon_pad; lon_max += lon_pad
    else:
        lat_min, lat_max, lon_min, lon_max = LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

    # ----- Grid -----
    lats = np.round(np.arange(lat_min, lat_max + 1e-9, STEP_DEG), 6)
    lons = np.round(np.arange(lon_min, lon_max + 1e-9, STEP_DEG), 6)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")

    # Optional cap setup (safe with None)
    vmax_cap = None
    if (CLIP_MAX_MULT is not None) and np.isfinite(v_hi):
        vmax_cap = float(v_hi) * float(CLIP_MAX_MULT)

    # ----- Predict & write -----
    pred_rows = []
    start, end = date(*PRED_START), date(*PRED_END)
    for d in daterange(start, end):
        dow = d.weekday()
        feats = np.stack([
            LAT.ravel().astype(np.float32),
            LON.ravel().astype(np.float32),
            np.full(LAT.size, d.year, dtype=np.float32),
            np.full(LAT.size, dow,   dtype=np.float32)
        ], axis=1)

        z = model.predict(feats, verbose=0).ravel().astype(np.float64)  # standardized preds
        vals = z * y_std + y_mean                                       # back to original units

        # Optional clamps (only applied if enabled)
        if CLIP_MIN is not None:
            vals = np.maximum(vals, CLIP_MIN)
        if vmax_cap is not None:
            vals = np.minimum(vals, vmax_cap)

        iso = d.isoformat()
        for (lat, lon, v) in zip(feats[:,0], feats[:,1], vals):
            pred_rows.append((iso, float(lat), float(lon), float(v)))

    out = pd.DataFrame(pred_rows, columns=["time","latitude","longitude","value"])
    out.to_csv(OUTPUT_CSV, index=False)

    # Sanity prints
    print(f"Wrote {len(out):,} rows → {OUTPUT_CSV}")
    print("train value stats:",
          f"min={np.min(y_raw):.3g} max={np.max(y_raw):.3g} mean={y_mean:.3g} std={y_std:.3g}")
    print("pred value stats:",
          f"min={out['value'].min():.3g} max={out['value'].max():.3g}")

if __name__ == "__main__":
    main()
