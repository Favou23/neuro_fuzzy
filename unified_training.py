# unified_training.py
import os
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xanfis import AnfisClassifier
from data_ingestion import load_and_prepare_data, prepare_duval_features
# from iec_rule_based import duval_polygons, duval_polygon_classify
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("encoders", exist_ok=True)

class AnfisWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, num_rules=5, epochs=25, batch_size=16, optim='Adam',
                 valid_rate=0.1, early_stopping=True, verbose=False, device=None):
        self.num_rules = num_rules
        self.epochs = epochs
        self.batch_size = batch_size
        self.optim = optim
        self.valid_rate = valid_rate
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.device = device
        self.model_ = None
        self.loss_history_ = None

    def fit(self, X, y):
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int32).reshape(-1)
        self.model_ = AnfisClassifier(
            num_rules=int(self.num_rules),
            epochs=int(self.epochs),
            batch_size=int(self.batch_size),
            optim=self.optim,
            valid_rate=self.valid_rate,
            early_stopping=self.early_stopping,
            verbose=self.verbose,
            device=self.device
        )
        self.model_.fit(X_np, y_np)
        if hasattr(self.model_, "loss_history_"):
            self.loss_history_ = self.model_.loss_history_
        return self

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("Model is not fitted yet.")
        X_np = np.asarray(X, dtype=np.float32)
        return self.model_.predict(X_np)

    def get_params(self, deep=True):
        return {
            "num_rules": self.num_rules,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "optim": self.optim,
            "valid_rate": self.valid_rate,
            "early_stopping": self.early_stopping,
            "verbose": self.verbose,
            "device": self.device
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

if __name__ == "__main__":
    methods = {
        "duval": (os.path.join(BASE_DIR, "duval_data_generator", "datasets", "duval_polygon_dataset.csv"), "FAULT"),
        # fill other dataset paths if you use them
    }

    param_grid = {
        'clf__num_rules': [5],  # reduce search for speed, tune as needed
        'clf__epochs': [25],
        'clf__batch_size': [16, 32],
        'clf__optim': ['Adam']
    }

    for method_name, (file_path, label_col) in methods.items():
        print(f"\n=== Training for {method_name} Method ===")
        try:
            # Enable IEC augmentation here
            X, y, le = load_and_prepare_data(file_path, label_col, method_name,
                                            augment_iec=True,
                                            iec_synth_per_class=300,
                                            iec_ppm_range=(10, 5000),
                                            iec_jitter=0.08,
                                            seed=42)
            print("Label distribution (encoded):")
            counts = pd.Series(y).map(lambda iv: le.inverse_transform([iv])[0]).value_counts()
            print(counts)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y
            )

            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=42, k_neighbors=5)),
                ('scaler', StandardScaler()),
                ('clf', AnfisWrapper())
            ])

            grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=1)
            grid.fit(X_train, y_train)

            print("Best Parameters:", grid.best_params_)
            best_pipeline = grid.best_estimator_

            y_pred = best_pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            print("Test Accuracy:", acc)
            print("Classification Report:")
            print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

            # Save the full pipeline (scaling + classifier)
            # pipeline_save_dir = os.path.join(RESULTS_DIR, f"{method_name}_pipeline")
            # os.makedirs(pipeline_save_dir, exist_ok=True)
            # pipeline_path = os.path.join(pipeline_save_dir, "pipeline.joblib")
            # AnfisClassifier.save_model(best_pipeline, pipeline_path)
            # print(f"Saved full pipeline at: {pipeline_path}")
            
            anfis_model = best_pipeline.named_steps['clf'].model_
            save_dir = os.path.join(RESULTS_DIR, f"{method_name}_anfis")
            save_path = os.path.join(save_dir, "model.pkl")
            anfis_model.save_model(save_path=save_path)
            print(f"Saved ANFIS model to: {save_path}")

            # Save the LabelEncoder
            encoder_path = os.path.join("encoders", f"{method_name}_label_encoder.pkl")
            joblib.dump(le, encoder_path)
            print(f"Saved LabelEncoder at: {encoder_path}")
            print("Classes:", list(le.classes_))

            # Plot confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{method_name} Confusion Matrix")
            plot_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{method_name.lower()}.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved confusion matrix at: {plot_path}")

            # Save loss history if available
            loss_history = best_pipeline.named_steps['clf'].loss_history_
            if loss_history is not None:
                plt.figure(figsize=(6,4))
                plt.plot(loss_history, marker="o")
                plt.title(f"{method_name} Loss Curve")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.grid(True)
                loss_path = os.path.join(RESULTS_DIR, f"loss_curve_{method_name.lower()}.png")
                plt.savefig(loss_path)
                plt.close()
                print(f"Saved loss curve at: {loss_path}")
            else:
                print("Loss history not available.")
        except Exception as e:
            print(f"Error in {method_name} pipeline: {e}")
            traceback.print_exc()
