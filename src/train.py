import os
import tempfile
import pandas as pd
import numpy as np
import mlflow
import mlflow.data
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
from sqlalchemy import func, select
from database import sync_engine
from src.models.neural_networks import DenguePredictorLSTM, DenguePredictorPLE
from src.models.caso_dengue import CasoDengue

_STATE_IBGE_CODE: dict[str, int] = {
    "RO": 11,
    "AC": 12,
    "AM": 13,
    "RR": 14,
    "PA": 15,
    "AP": 16,
    "TO": 17,
    "MA": 21,
    "PI": 22,
    "CE": 23,
    "RN": 24,
    "PB": 25,
    "PE": 26,
    "AL": 27,
    "SE": 28,
    "BA": 29,
    "MG": 31,
    "ES": 32,
    "RJ": 33,
    "SP": 35,
    "PR": 41,
    "SC": 42,
    "RS": 43,
    "MS": 50,
    "MT": 51,
    "GO": 52,
    "DF": 53,
}

# =============================================================================
# 1. CONFIGURATION AND UTILITIES
# =============================================================================


def experiments_settings(state: str):
    """Defines experiment settings for both LSTM and PLE models."""
    experiment_description = (
        f"Model comparison (LSTM vs PLE) for dengue forecasting in {state}."
    )
    experiment_tags = {
        "project_name": "dengue-forecasting-comparison",
        "state": state,
        "team": "leggen-assis-ml",
        "project_quarter": "Q2-2026",
        "mlflow.note.content": experiment_description,
    }

    lstm_search_space = {
        "model_type": ["LSTM"],
        "learning_rate": [0.001, 0.0001],
        "hidden_size": [64, 128],
        "n_layers": [2],
        "sequence_length": [7, 14],
        "dropout": [0.2],
    }
    ple_search_space = {
        "model_type": ["PLE"],
        "learning_rate": [0.001, 0.0001],
        "hidden_size": [64, 128],
        "num_experts": [3, 4],
        "num_ple_layers": [2],
        "sequence_length": [7, 14],
        "dropout": [0.2],
    }

    base_params = {"epochs": [100], "batch_size": [64]}

    lstm_param_grid = list(ParameterGrid({**lstm_search_space, **base_params}))
    ple_param_grid = list(ParameterGrid({**ple_search_space, **base_params}))

    param_grid = lstm_param_grid + ple_param_grid

    return {"tags": experiment_tags, "param_grid": param_grid}


def create_sequences(data, seq_length):
    """Creates sliding-window sequences for time series forecasting."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i : (i + seq_length)])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)


def calculate_metrics(y_true, y_pred):
    """Computes regression metrics: RMSE, MAE, MAPE, and R²."""
    y_true = np.nan_to_num(y_true, nan=0.0)
    y_pred = np.nan_to_num(y_pred, nan=0.0)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    if np.var(y_true) < 1e-8:
        r2 = 1.0 if np.allclose(y_true, y_pred, atol=1e-4) else 0.0
    else:
        r2 = r2_score(y_true, y_pred)

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


# =============================================================================
# 2. DATA PREPARATION
# =============================================================================


def fetch_and_clean_data(state: str):
    """Fetches daily case counts from the database and applies initial cleaning."""
    ibge_code = _STATE_IBGE_CODE.get(state.upper())
    if ibge_code is None:
        print(f"Unknown state abbreviation: {state}. Skipping.")
        return None

    print(f"Connecting to database for state {state} (IBGE code: {ibge_code})...")
    with sync_engine.connect() as conn:
        query = (
            select(
                CasoDengue.notification_date,
                func.count().label("daily_cases"),
            )
            .where(CasoDengue.state_ibge_code == ibge_code)
            .group_by(CasoDengue.notification_date)
            .order_by(CasoDengue.notification_date)
        )
        result = conn.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    if df.empty:
        print(f"No data found for state {state}. Skipping.")
        return None

    df = df.rename(columns={"notification_date": "date"})

    print(f"Records before cleaning: {len(df)}")

    q99 = df["daily_cases"].quantile(0.99)
    print(f"Q99: {q99:.2f}")

    df = df[df["daily_cases"] <= q99 * 3]
    print(f"Records after removing extreme outliers: {len(df)}")

    return df


def prepare_data_for_run(df: pd.DataFrame, seq_length: int):
    """Scales data, creates sequences, and splits into train/validation sets."""
    time_series_raw = df["daily_cases"].values.astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    time_series_scaled = scaler.fit_transform(time_series_raw.reshape(-1, 1))

    X, y = create_sequences(time_series_scaled, seq_length)

    if len(X) == 0:
        print("Could not create sequences. Skipping.")
        return None, None

    print(f"\nTotal sequences created: {len(X)}")

    train_size = int(len(X) * 0.8)
    original_y_full = time_series_raw[seq_length:]

    y_val_proposed = y[train_size:]
    val_non_zero = np.count_nonzero(y_val_proposed)
    val_zero_ratio = 1 - (val_non_zero / len(y_val_proposed))

    if val_zero_ratio > 0.9:
        print(
            f"\nTemporal split would result in {val_zero_ratio * 100:.1f}% zeros in validation!"
        )
        print("   Falling back to stratified random split.")

        y_bins = np.digitize(
            y.flatten(), bins=np.percentile(y.flatten(), [0, 25, 50, 75, 100])
        )

        X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
            X, y, np.arange(len(X)), test_size=0.2, stratify=y_bins, random_state=42
        )

        original_train_y = original_y_full[idx_train]
        original_val_y = original_y_full[idx_val]
        print("   Stratified split applied successfully.")
    else:
        print("Using standard temporal split (80/20).")
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        original_train_y = original_y_full[:train_size]
        original_val_y = original_y_full[train_size:]

    print("\nDEBUG - Scale check (Val):")
    print(
        f"  y_val scaled: min={y_val.min():.6f}, max={y_val.max():.6f}, mean={y_val.mean():.6f}"
    )
    print(
        f"  original_val_y: min={original_val_y.min():.2f}, max={original_val_y.max():.2f}, mean={original_val_y.mean():.2f}"
    )

    if original_val_y.max() < 1.0:
        print("\nWARNING: Validation data has very low values!")

    if np.all(original_val_y == 0):
        print("\nCRITICAL ERROR: All validation values are zero! Skipping...")
        return None, None

    data_dict = {
        "X_train_tensor": torch.from_numpy(X_train).float(),
        "y_train_tensor": torch.from_numpy(y_train).float(),
        "X_val_tensor": torch.from_numpy(X_val).float(),
        "y_val_tensor": torch.from_numpy(y_val).float(),
        "original_train_y": original_train_y,
        "original_val_y": original_val_y,
    }

    print(
        f"Shapes — X_train: {data_dict['X_train_tensor'].shape}, y_train: {data_dict['y_train_tensor'].shape}"
    )
    print(
        f"Shapes — X_val: {data_dict['X_val_tensor'].shape}, y_val: {data_dict['y_val_tensor'].shape}"
    )

    return data_dict, scaler


# =============================================================================
# 3. TRAINING FUNCTIONS
# =============================================================================


def instantiate_model(params: dict, device):
    """Instantiates an LSTM or PLE model based on the given parameter dict."""
    model_type = params["model_type"]

    if model_type == "LSTM":
        model = DenguePredictorLSTM(
            n_features=1,
            hidden_size=params["hidden_size"],
            n_layers=params["n_layers"],
            dropout=params.get("dropout", 0.0),
        ).to(device)
    elif model_type == "PLE":
        model = DenguePredictorPLE(
            n_features=1,
            hidden_size=params["hidden_size"],
            num_experts=params["num_experts"],
            num_layers=params["num_ple_layers"],
            dropout=params.get("dropout", 0.0),
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Runs the training loop for a single epoch."""
    model.train()
    epoch_train_loss = 0.0
    num_batches = 0

    for seqs, labels in train_loader:
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_train_loss += loss.item()
        num_batches += 1

    return epoch_train_loss / num_batches


def perform_validation(model, criterion, device, scaler, data_dict):
    """Runs the validation step and computes RMSE, MAE, MAPE, and R² metrics."""
    model.eval()
    with torch.no_grad():
        X_train_tensor = data_dict["X_train_tensor"].to(device)
        y_train_tensor = data_dict["y_train_tensor"].to(device)
        X_val_tensor = data_dict["X_val_tensor"].to(device)
        y_val_tensor = data_dict["y_val_tensor"].to(device)
        original_train_y = data_dict["original_train_y"]
        original_val_y = data_dict["original_val_y"]

        val_outputs_scaled = model(X_val_tensor)
        val_loss = criterion(val_outputs_scaled, y_val_tensor).item()

        train_outputs_scaled = model(X_train_tensor)
        train_loss = criterion(train_outputs_scaled, y_train_tensor).item()

        y_train_pred = scaler.inverse_transform(
            train_outputs_scaled.cpu().numpy().reshape(-1, 1)
        ).flatten()

        y_val_pred = scaler.inverse_transform(
            val_outputs_scaled.cpu().numpy().reshape(-1, 1)
        ).flatten()

        train_metrics = calculate_metrics(original_train_y, y_train_pred)
        val_metrics = calculate_metrics(original_val_y, y_val_pred)

    return val_loss, train_loss, train_metrics, val_metrics


def log_final_artifacts(model, scaler, run_name: str, data_dict: dict):
    """Generates diagnostic plots and logs all artifacts (model, scaler, plots) to MLflow."""
    print("\nGenerating plots and saving final artifacts...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = data_dict["X_train_tensor"].to(device)
    X_val_tensor = data_dict["X_val_tensor"].to(device)
    y_train_true_final = data_dict["original_train_y"]
    y_val_true_final = data_dict["original_val_y"]

    model.eval()
    with torch.no_grad():
        final_train_pred = model(X_train_tensor).cpu().numpy().reshape(-1, 1)
        final_val_pred = model(X_val_tensor).cpu().numpy().reshape(-1, 1)

    y_train_pred_final = scaler.inverse_transform(final_train_pred).flatten()
    y_val_pred_final = scaler.inverse_transform(final_val_pred).flatten()

    print(
        f"Final stats (Val) — Pred min: {y_val_pred_final.min():.2f}, Pred max: {y_val_pred_final.max():.2f}"
    )
    print(
        f"Final stats (Val) — True min: {y_val_true_final.min():.2f}, True max: {y_val_true_final.max():.2f}"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        scaler_path = os.path.join(tmp_dir, "scaler.pkl")
        joblib.dump(scaler, scaler_path)

        fig_ts, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        sample_size = min(500, len(y_train_true_final))
        sample_indices = np.linspace(
            0, len(y_train_true_final) - 1, sample_size, dtype=int
        )

        ax1.plot(
            sample_indices,
            y_train_true_final[sample_indices],
            label="Actual",
            alpha=0.7,
            lw=1.5,
        )
        ax1.plot(
            sample_indices,
            y_train_pred_final[sample_indices],
            label="Prediction",
            linestyle="--",
            alpha=0.7,
            lw=1.5,
        )
        ax1.set_title(f"Training Set: Predictions vs. Actuals — {run_name}")
        ax1.set_xlabel("Sample Index")
        ax1.set_ylabel("Daily Cases")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        sample_size_val = min(200, len(y_val_true_final))
        sample_indices_val = np.linspace(
            0, len(y_val_true_final) - 1, sample_size_val, dtype=int
        )

        ax2.plot(
            sample_indices_val,
            y_val_true_final[sample_indices_val],
            label="Actual",
            alpha=0.7,
            lw=1.5,
        )
        ax2.plot(
            sample_indices_val,
            y_val_pred_final[sample_indices_val],
            label="Prediction",
            linestyle="--",
            alpha=0.7,
            lw=1.5,
        )
        ax2.set_title(f"Validation Set: Predictions vs. Actuals — {run_name}")
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Daily Cases")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path_ts = os.path.join(tmp_dir, "predictions_vs_reals.png")
        fig_ts.savefig(plot_path_ts, dpi=100)
        plt.close(fig_ts)

        fig_sc, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.scatter(
            y_train_true_final[sample_indices],
            y_train_pred_final[sample_indices],
            alpha=0.5,
            s=20,
        )
        max_val = max(y_train_true_final.max(), y_train_pred_final.max())
        min_val = min(y_train_true_final.min(), y_train_pred_final.min())
        ax1.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )
        ax1.set_xlabel("Actual Values")
        ax1.set_ylabel("Predicted Values")
        ax1.set_title("Training Set - Scatter Plot")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.scatter(
            y_val_true_final[sample_indices_val],
            y_val_pred_final[sample_indices_val],
            alpha=0.5,
            s=20,
            color="orange",
        )
        max_val_v = max(y_val_true_final.max(), y_val_pred_final.max())
        min_val_v = min(y_val_true_final.min(), y_val_pred_final.min())
        ax2.plot(
            [min_val_v, max_val_v],
            [min_val_v, max_val_v],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )
        ax2.set_xlabel("Actual Values")
        ax2.set_ylabel("Predicted Values")
        ax2.set_title("Validation Set - Scatter Plot")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path_sc = os.path.join(tmp_dir, "scatter_plot.png")
        fig_sc.savefig(plot_path_sc, dpi=100)
        plt.close(fig_sc)

        mlflow.pytorch.log_model(model, artifact_path="model")
        mlflow.log_artifact(scaler_path, artifact_path="preprocessor")
        mlflow.log_artifact(plot_path_ts, artifact_path="plots")
        mlflow.log_artifact(plot_path_sc, artifact_path="plots")


# =============================================================================
# 4. MAIN TRAINING FUNCTION
# =============================================================================


def train_model(params: dict, state: str, data_dict: dict, scaler, dataset):
    """Orchestrates training for a single MLflow run."""

    model_type = params["model_type"]
    if model_type == "LSTM":
        run_name = (
            f"LSTM_LR_{params['learning_rate']}_HS_{params['hidden_size']}_"
            f"NL_{params['n_layers']}_SEQ_{params['sequence_length']}"
        )
    else:
        run_name = (
            f"PLE_LR_{params['learning_rate']}_HS_{params['hidden_size']}_"
            f"E_{params['num_experts']}_SEQ_{params['sequence_length']}"
        )

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"\n--- Starting run ({model_type}): {run_name} ---")

        mlflow.log_input(dataset=dataset, context="training")
        mlflow.set_tags(
            {
                "model_type": model_type,
                "Optimizer": "Adam",
                "run_purpose": "Grid Search Comparison",
                "target_variable": "daily_cases",
                "feature_set": "v_univariate",
                "developer": "leggen-assis",
                "data_source_table": "casos_dengue",
                "data_source_filter": f"state='{state}'",
            }
        )
        mlflow.log_params(params)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        X_train_tensor = data_dict["X_train_tensor"].to(device)
        y_train_tensor = data_dict["y_train_tensor"].to(device)

        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=params["batch_size"],
            shuffle=True,
        )

        model = instantiate_model(params, device)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=False
        )

        print("Starting model training...")
        best_val_loss = float("inf")
        patience_counter = 0
        early_stopping_patience = 20
        best_model_state = model.state_dict().copy()

        for epoch in range(params["epochs"]):
            avg_train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )

            val_loss, train_loss, train_metrics, val_metrics = perform_validation(
                model, criterion, device, scaler, data_dict
            )

            mlflow.log_metric("train_loss_batch_avg", avg_train_loss, step=epoch)
            mlflow.log_metric("train_loss_full", train_loss, step=epoch)
            mlflow.log_metric("val_loss_scaled", val_loss, step=epoch)
            mlflow.log_metric(
                "learning_rate", optimizer.param_groups[0]["lr"], step=epoch
            )

            for key, val in train_metrics.items():
                mlflow.log_metric(f"train_{key}", val, step=epoch)
            for key, val in val_metrics.items():
                mlflow.log_metric(f"val_{key}", val, step=epoch)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{params['epochs']}] | "
                    f"Train Loss: {avg_train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Val RMSE: {val_metrics['rmse']:.2f} | "
                    f"Val R2: {val_metrics['r2']:.3f}"
                )

        print("\nTraining complete. Loading best model state...")
        model.load_state_dict(best_model_state)

        log_final_artifacts(model, scaler, run_name, data_dict)

        print(f"Model and artifacts saved for run: {run_id}")


# =============================================================================
# 5. EXPERIMENT ORCHESTRATION
# =============================================================================


def run_experiments(state: str):
    """Orchestrates the full grid-search experiment pipeline for a single state."""
    print(f"\n{'=' * 60}\nStarting experiment for state: {state}\n{'=' * 60}")

    settings = experiments_settings(state)

    df = fetch_and_clean_data(state)
    if df is None:
        return

    mlflow_dataset = mlflow.data.from_pandas(
        df, name=f"casos_dengue_{state.lower()}_train"
    )
    experiment_name = f"Dengue Forecasting Comparison - {state}"
    mlflow.set_experiment(experiment_name)
    mlflow.set_experiment_tags(settings["tags"])

    print(f"\nStarting grid search with {len(settings['param_grid'])} combinations...")

    for idx, params in enumerate(settings["param_grid"], 1):
        print(f"\n{'=' * 60}")
        print(f"[{idx}/{len(settings['param_grid'])}] Testing combination: {params}")
        print(f"{'=' * 60}")

        data_dict, scaler = prepare_data_for_run(df, params["sequence_length"])

        if data_dict is None:
            continue

        train_model(params, state, data_dict, scaler, mlflow_dataset)

    print(f"\n{'=' * 60}")
    print(f"All experiments for state {state} completed!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    states_to_train = ["CE"]

    for state_code in states_to_train:
        run_experiments(state=state_code)

    print(f"\n{'=' * 60}\n Process completed successfully!\n{'=' * 60}")
