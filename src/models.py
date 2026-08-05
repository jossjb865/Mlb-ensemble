#!/usr/bin/env python3
"""Ensemble completo: XGB + CatBoost + 3 LSTM + NN."""
from __future__ import annotations
import os
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, mean_absolute_error
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import joblib
from features import FEATURE_COLS

def prepare_xy(df: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLS].astype(np.float32).values
    y = df[target].values
    return X, y

def train_xgb(X_train, y_train, X_val, y_val, task: str = "class"):
    if task == "class":
        model = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            eval_metric="logloss", early_stopping_rounds=40,
            random_state=42, n_jobs=-1
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            early_stopping_rounds=40, random_state=42, n_jobs=-1
        )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model

def train_catboost(X_train, y_train, X_val, y_val, task: str = "class"):
    if task == "class":
        model = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
            eval_metric="Logloss", early_stopping_rounds=40,
            random_seed=42, verbose=False
        )
    else:
        model = CatBoostRegressor(
            iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
            eval_metric="RMSE", early_stopping_rounds=40,
            random_seed=42, verbose=False
        )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    return model

def build_lstm(input_shape: Tuple[int, int], task: str = "class") -> tf.keras.Model:
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.30)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(16, activation="relu")(x)
    if task == "class":
        out = layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
    else:
        out = layers.Dense(1)(x)
        loss = "mse"
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss)
    return model

def sequence_data(X: np.ndarray, y: np.ndarray, seq_len: int = 5):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys)

def train_lstm(X_train, y_train, X_val, y_val, task: str, seed: int):
    tf.random.set_seed(seed)
    seq_len = 5
    Xtr, ytr = sequence_data(X_train, y_train, seq_len)
    Xva, yva = sequence_data(X_val, y_val, seq_len)
    if len(Xtr) < 60:
        return None
    model = build_lstm((seq_len, X_train.shape[1]), task)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=60, batch_size=32, callbacks=[es], verbose=0)
    return model

def train_nn(X_train, y_train, X_val, y_val, task: str):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)
    inp = layers.Input(shape=(X_train.shape[1],))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(32, activation="relu")(x)
    if task == "class":
        out = layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
    else:
        out = layers.Dense(1)(x)
        loss = "mse"
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    model.fit(Xtr, y_train, validation_data=(Xva, y_val), epochs=80, batch_size=64, callbacks=[es], verbose=0)
    return model, scaler

def evaluate_class(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    return {"accuracy": float(accuracy_score(y_true, pred)),
            "logloss": float(log_loss(y_true, np.clip(proba, 1e-7, 1-1e-7)))}

def evaluate_reg(y_true, pred):
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
            "mae": float(mean_absolute_error(y_true, pred))}

def train_ensemble(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    os.makedirs("models", exist_ok=True)
    results = {}
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    for target, task in [("home_win", "class"), ("total_runs", "reg")]:
        print(f"=== {target} ===")
        X_tr, y_tr = prepare_xy(train_df, target)
        X_te, y_te = prepare_xy(test_df, target)
        val_split = int(len(X_tr) * 0.85)
        X_train, y_train = X_tr[:val_split], y_tr[:val_split]
        X_val, y_val = X_tr[val_split:], y_tr[val_split:]

        models_dict = {}
        m = train_xgb(X_train, y_train, X_val, y_val, task)
        models_dict["xgb"] = m
        joblib.dump(m, f"models/{target}_xgb.joblib")

        m = train_catboost(X_train, y_train, X_val, y_val, task)
        models_dict["cat"] = m
        joblib.dump(m, f"models/{target}_cat.joblib")

        nn, scaler = train_nn(X_train, y_train, X_val, y_val, task)
        models_dict["nn"] = (nn, scaler)
        nn.save(f"models/{target}_nn.keras")
        joblib.dump(scaler, f"models/{target}_nn_scaler.joblib")

        for name, seed in [("lstm_momentum", 11), ("lstm_result", 22), ("lstm_model", 33)]:
            m = train_lstm(X_train, y_train, X_val, y_val, task, seed)
            if m is not None:
                models_dict[name] = m
                m.save(f"models/{target}_{name}.keras")

        preds = []
        for name, m in models_dict.items():
            if name == "nn":
                nn_m, sc = m
                p = nn_m.predict(sc.transform(X_te), verbose=0).ravel()
            elif name.startswith("lstm"):
                seq_len = 5
                if len(X_te) <= seq_len:
                    continue
                Xseq, _ = sequence_data(X_te, y_te, seq_len)
                p = m.predict(Xseq, verbose=0).ravel()
                p = np.concatenate([np.full(seq_len, np.nan), p])
            else:
                p = m.predict_proba(X_te)[:, 1] if task == "class" else m.predict(X_te)
            if len(p) == len(y_te):
                preds.append(p)

        ensemble = np.nanmean(np.vstack(preds), axis=0) if preds else np.zeros(len(y_te))
        metrics = evaluate_class(y_te, ensemble) if task == "class" else evaluate_reg(y_te, ensemble)
        results[target] = metrics
        print(metrics)

        test_out = test_df.copy()
        test_out[f"pred_{target}"] = ensemble
        test_out.to_csv(f"data/test_{target}.csv", index=False)

    return results

if __name__ == "__main__":
    df = pd.read_parquet("data/features.parquet")
    print(train_ensemble(df))
