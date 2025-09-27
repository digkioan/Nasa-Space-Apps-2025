# make_predictions_no2.py
# Train a TF neural net and export predictions to CSV: time,latitude,longitude,value

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from datetime import date, timedelta

# ===================== CONFIG =====================
CSV_HISTORY = "no2_temp_data.csv"        # your merged actuals (same 4 columns)
OUTPUT_CSV  = "no2_prediction.csv"     # write here (serve this path on your site)
RANDOM_SEED = 42

# Prediction window (inclusive)
PRED_YEAR = 2025                        # change as needed
PRED_START = (PRED_YEAR, 10, 24)        # Oct 24
PRED_END   = (PRED_YEAR, 10, 30)        # Oct 30

# Prediction grid over Greece (adjust to your coverage)
LAT_MIN, LAT_MAX = 34.0, 42.0
LON_MIN, LON_MAX = 18.0, 28.0
STEP_DEG = 0.2                          # 0.2° ~ 20–22 km; smaller = denser grid

EPOCHS = 200
BATCH  = 64
# ==================================================

def daterange(d0: date, d1: date):
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)

def build_model(input_dim: int, normalizer: keras.layers.Normalization):
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = normalizer(inputs)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def main():
    # ---------- Load & features ----------
    df = pd.read_csv(CSV_HISTORY)
    for c in ["time","latitude","longitude","value"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time","latitude","longitude","value"]).copy()
    df["year"] = df["time"].dt.year.astype(int)
    df["day_of_week"] = df["time"].dt.dayofweek.astype(int)

    feature_cols = ["latitude","longitude","year","day_of_week"]
    X = df[feature_cols].to_numpy(np.float32)
    y = df["value"].to_numpy(np.float32)

    # ---------- Split (robust) ----------
    years = sorted(df["year"].unique())
    if len(years) >= 2:
        last_year = years[-1]
        train_df = df[df["year"] < last_year].copy()
        if train_df.empty:
            # safety: random 80/20
            train_df = df.sample(frac=0.8, random_state=RANDOM_SEED)
            val_df   = df.drop(train_df.index).copy()
        else:
            val_df   = df[df["year"] == last_year].copy()
    else:
        # single-year dataset → random 80/20
        train_df = df.sample(frac=0.8, random_state=RANDOM_SEED)
        val_df   = df.drop(train_df.index).copy()

    X_train = train_df[feature_cols].to_numpy(np.float32)
    y_train = train_df["value"].to_numpy(np.float32)
    X_val   = val_df[feature_cols].to_numpy(np.float32)
    y_val   = val_df["value"].to_numpy(np.float32)

    # ---------- Normalizer ----------
    norm = keras.layers.Normalization(axis=-1)
    norm.adapt(X_train)

    # ---------- Model ----------
    model = build_model(X_train.shape[1], norm)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    ]

    # Use validation only if we have some
    has_val = len(X_val) > 0
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if has_val else None,
        epochs=EPOCHS, batch_size=BATCH, verbose=1, shuffle=True
    )

    # ---------- Build prediction grid ----------
    start = date(*PRED_START)
    end   = date(*PRED_END)
    lats = np.round(np.arange(LAT_MIN, LAT_MAX + 1e-6, STEP_DEG), 6)
    lons = np.round(np.arange(LON_MIN, LON_MAX + 1e-6, STEP_DEG), 6)

    pred_rows = []
    for d in daterange(start, end):
        dow = d.weekday()  # 0=Mon..6=Sun
        # Build features for all grid points for this date
        LAT, LON = np.meshgrid(lats, lons, indexing="ij")
        feats = np.stack([
            LAT.ravel().astype(np.float32),
            LON.ravel().astype(np.float32),
            np.full(LAT.size, d.year, dtype=np.float32),
            np.full(LAT.size, dow, dtype=np.float32)
        ], axis=1)
        vals = model.predict(feats, verbose=0).ravel()

        # Append rows
        iso = d.isoformat()
        for (lat, lon, v) in zip(feats[:,0], feats[:,1], vals):
            pred_rows.append((iso, float(lat), float(lon), float(v)))

    out = pd.DataFrame(pred_rows, columns=["time","latitude","longitude","value"])
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(out):,} rows → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
