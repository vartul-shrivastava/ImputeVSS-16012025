import os
import io
import json
import base64
import re
import subprocess 
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, make_response, session
from scipy.stats import gaussian_kde, skew, kurtosis, ks_2samp
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import LabelEncoder

import ollama  # Ensure the Ollama Python package is installed
from bs4 import BeautifulSoup

# -------------------------------------------------
# Flask Setup and Directories
# -------------------------------------------------
app = Flask(__name__)
app.secret_key = "your_default_secret_key"  # Replace with a secure key

UPLOAD_FOLDER = "uploads"
PIPELINE_SAVE_DIR = "pipeline_data"
PROJECTS_SAVE_DIR = "projects"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PIPELINE_SAVE_DIR, exist_ok=True)
os.makedirs(PROJECTS_SAVE_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------------------------------------
# Global In-Memory Data Store
# -------------------------------------------------
data_store = {
    "original_data": None,        # The active/working DataFrame
    "imputed_versions": {},       # { column: { method: pd.Series } } (temporary imputed copies)
    "imputed_stats": {},          # { column: { method: stats_dict } }
    "pipeline_steps": []          # Pipeline steps (if needed)
}
custom_prompt = None
default_prompt = """You are analyzing an imputation statistics table created by a state-of-the-art data imputation toolkit.
Here is the provided table for evaluation: {table}
Based on the metrics provided (mean, median, std, KDE overlap, skew, kurtosis, KS statistic, KS p-value, and KL divergence for numeric columns, or mode, unique count, and mode frequency for categorical columns), generate a detailed AI-based summary that recommends the best imputation technique. Explain your reasoning step-by-step and discuss the strengths and weaknesses of each method.
"""

# -------------------------------------------------
# Prompt Management Endpoints
# -------------------------------------------------
@app.route('/reset_prompt', methods=['POST'])
def reset_prompt():
    global custom_prompt
    try:
        custom_prompt = None  # Clear the custom prompt
        return jsonify({
            'status': 'success',
            'message': 'Prompt reset to default successfully.',
            'default_prompt': default_prompt
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/get_current_prompt', methods=['GET'])
def get_current_prompt():
    global custom_prompt
    try:
        current_prompt = custom_prompt or default_prompt
        return jsonify({'prompt': current_prompt})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/update_prompt', methods=['POST'])
def update_prompt():
    global custom_prompt
    data = request.get_json()
    custom_prompt = data.get('prompt')
    return jsonify({'status': 'success', 'message': 'Prompt updated successfully'})

def is_ollama_running():
    """
    Checks if Ollama is running by attempting to execute 'ollama list'.
    Returns True if Ollama responds, False otherwise.
    """
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Return output as string
            timeout=5  # Timeout after 5 seconds
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        return False

# Route to check AI readiness and available models
@app.route('/check_ai_readiness', methods=['GET'])
def check_ai_readiness():
    if not is_ollama_running():
        return jsonify({
            "ollama_ready": False,
            "models": [],
            "error": "Ollama is not running or not found in PATH."
        })

    try:
        # Fetch available models from Ollama
        model_data = str(ollama.list())  # Assume this returns the list of Model objects

        # Regular expression to match the model name
        pattern = r"model='(.*?)'"  # Captures content between model=' and '

        # Use re.findall to extract all matches
        models = re.findall(pattern, model_data)
        models = [name.strip() for name in models if name.strip()]  # Strip whitespace and filter out empty strings

        return jsonify({
            "ollama_ready": True,
            "models": models
        })
    except Exception as e:
        return jsonify({
            "ollama_ready": True,
            "models": [],
            "error": f"Error fetching Ollama models: {e}"
        })


# -------------------------------------------------
# Helper Functions (Imputation, KDE, Stats, etc.)
# -------------------------------------------------
def compute_kde(values):
    if len(values) < 2:
        return [], []
    kde_func = gaussian_kde(values)
    kde_x = np.linspace(min(values), max(values), 100)
    kde_y = kde_func(kde_x)
    return kde_x.tolist(), kde_y.tolist()


def compute_comparative_stats(original_vals, imputed_vals, orig_kde, imp_kde):
    # Compute statistics for numeric columns only.
    def safe_stats(arr):
        from numpy import mean, median, std
        if len(arr) == 0:
            return None, None, None, None, None
        return (float(mean(arr)), float(median(arr)), float(std(arr)), float(skew(arr)), float(kurtosis(arr)))
    orig_mean, orig_med, orig_std, orig_skew, orig_kurt = safe_stats(original_vals)
    imp_mean, imp_med, imp_std, imp_skew, imp_kurt = safe_stats(imputed_vals)
    x_orig, y_orig = orig_kde
    x_imp, y_imp = imp_kde
    overlap_value = None
    kl_divergence = None
    if len(x_orig) == len(x_imp) and len(x_orig) > 1:
        overlap_value = float(np.trapz(np.minimum(y_orig, y_imp), x_orig))
        y_orig_norm = y_orig / np.trapz(y_orig, x_orig)
        y_imp_norm = y_imp / np.trapz(y_imp, x_orig)
        kl_divergence = float(np.sum(y_orig_norm * np.log(y_orig_norm / y_imp_norm)))
    ks_stat, ks_pvalue = None, None
    if len(original_vals) > 1 and len(imputed_vals) > 1:
        stat_val, p_val = ks_2samp(original_vals, imputed_vals)
        ks_stat, ks_pvalue = float(stat_val), float(p_val)
    return {
        "original_mean": orig_mean,
        "imputed_mean": imp_mean,
        "original_median": orig_med,
        "imputed_median": imp_med,
        "original_std": orig_std,
        "imputed_std": imp_std,
        "kde_overlap": overlap_value,
        "original_skew": orig_skew,
        "imputed_skew": imp_skew,
        "original_kurtosis": orig_kurt,
        "imputed_kurtosis": imp_kurt,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "kl_divergence": kl_divergence
    }

def compute_categorical_stats(original_vals, imputed_vals):
    """Compute categorical statistics: mode, unique count, mode frequency, and unique percentage."""
    def get_stats(vals):
        series = pd.Series(vals)
        total = len(series)
        if series.empty or total == 0:
            return {"mode": None, "unique_count": 0, "mode_frequency": 0, "unique_percentage": 0}
        mode_val = series.mode().iloc[0] if not series.mode().empty else None
        unique_count = series.nunique()
        mode_freq = int((series == mode_val).sum()) if mode_val is not None else 0
        unique_percentage = (unique_count / total) * 100 if total > 0 else 0
        return {
            "mode": mode_val,
            "unique_count": unique_count,
            "mode_frequency": mode_freq,
            "unique_percentage": unique_percentage
        }

    orig_stats = get_stats(original_vals)
    imp_stats = get_stats(imputed_vals)
    return {"original": orig_stats, "imputed": imp_stats}

def apply_imputation(column_name, method, constant_value):
    df = data_store["original_data"].copy()
    is_numeric = pd.api.types.is_numeric_dtype(df[column_name])
    numeric_only = ["mean", "median"]

    if not is_numeric and method in numeric_only:
        raise ValueError(f"Method '{method}' is only for numeric columns, but '{column_name}' is categorical.")

    if method == "mean":
        return df[column_name].fillna(df[column_name].mean())
    elif method == "median":
        return df[column_name].fillna(df[column_name].median())
    elif method == "mode":
        mode_val = df[column_name].mode(dropna=True)
        if len(mode_val) == 0:
            raise ValueError(f"No valid mode found for '{column_name}'.")
        return df[column_name].fillna(mode_val[0])
    elif method == "constant":
        if constant_value is None or str(constant_value).strip() == "":
            raise ValueError("A constant value must be provided for constant imputation.")

        if is_numeric:
            try:
                constant_numeric = float(constant_value)
                return df[column_name].fillna(constant_numeric)
            except ValueError:
                raise ValueError("A numeric constant value must be provided for numeric columns.")
        else:
            return df[column_name].fillna(constant_value)
    elif method in ["knn", "mice"]:
        # Instead of imputing a single column, impute the entire dataset:
        imputed_df = impute_with_sklearn_dataset_with_labelencoder(df, method)
        # Return only the imputed column.
        return imputed_df[column_name]
    elif method == "complete-case":
        return df[column_name].dropna()
    elif method == "random":
        non_missing = df[column_name].dropna()
        if non_missing.empty:
            raise ValueError(f"No non-missing values available in column '{column_name}' for random imputation.")
        return df[column_name].apply(lambda x: x if pd.notna(x) else non_missing.sample(1).iloc[0])
    else:
        raise ValueError(f"Unknown imputation method: {method}")

def impute_with_sklearn(df, column_name, method, is_numeric):
    """
    Impute a single column using sklearn's imputation methods, but using the entire dataset 
    as context. The function first converts categorical variables to numeric using LabelEncoder,
    applies the imputer (KNN or MICE) to the full dataset, and then decodes any encoded columns.
    
    Parameters:
      - df: The DataFrame containing the data.
      - column_name: The column to be imputed.
      - method: The imputation method ('knn' or 'mice').
      - is_numeric: Boolean indicating if the column is numeric.
      
    Returns:
      - A pandas Series with the imputed values for column_name.
    """
    # Create a copy of the dataframe to impute on
    impute_df = df.copy()

    # Identify categorical columns that need encoding (including the target column if not numeric)
    categorical_cols = impute_df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Prepare a dictionary to store the encoders
    label_encoders = {}

    # Encode the categorical columns using LabelEncoder.
    # Note: even if column_name is not categorical, encoding it won't harm as it will be numeric.
    for col in categorical_cols:
        le = LabelEncoder()
        impute_df[col] = le.fit_transform(impute_df[col].astype(str))
        label_encoders[col] = le

    # For the target column, if it's non-numeric but marked as numeric by is_numeric=False,
    # force conversion to numeric (with errors coerced) so that the imputer works.
    if not is_numeric and column_name not in categorical_cols:
        impute_df[column_name] = pd.to_numeric(impute_df[column_name], errors='coerce')

    # Select the imputer
    if method == "knn":
        imputer = KNNImputer(n_neighbors=5)
    elif method == "mice":
        imputer = IterativeImputer(max_iter=10, random_state=0)
    else:
        raise ValueError(f"Method '{method}' not supported in impute_with_sklearn.")

    # Apply the imputer to the entire DataFrame.
    imputed_array = imputer.fit_transform(impute_df)
    imputed_df = pd.DataFrame(imputed_array, columns=impute_df.columns, index=impute_df.index)

    # For every originally categorical column, round the values and then decode back to their original labels.
    for col in categorical_cols:
        le = label_encoders[col]
        imputed_df[col] = imputed_df[col].round().astype(int)
        imputed_df[col] = le.inverse_transform(imputed_df[col])

    # Return the imputed target column as a Series.
    return imputed_df[column_name]

def impute_with_sklearn_dataset_with_labelencoder(df, method):
    """
    Impute the entire dataset using sklearn's imputation methods.
    This function encodes all columns so that every column is numeric.
    Then, the chosen imputer (KNN or MICE) is applied to the entire DataFrame.
    Finally, the originally categorical columns are decoded back to their original labels.
    
    Parameters:
      - df: The complete DataFrame.
      - method: The imputation method ('knn' or 'mice').
      
    Returns:
      - A new DataFrame with imputed values based on the analysis of the entire dataset.
    """
    # Create a copy of the entire dataset.
    impute_df = df.copy()
    
    # Identify all columns and determine which are categorical.
    categorical_cols = impute_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Dictionary to hold label encoders.
    label_encoders = {}
    
    # Encode each categorical column.
    for col in categorical_cols:
        le = LabelEncoder()
        impute_df[col] = le.fit_transform(impute_df[col].astype(str))
        label_encoders[col] = le
    
    # Do NOT subset; use the entire DataFrame for imputation.
    if method == "knn":
        imputer = KNNImputer(n_neighbors=5)
    elif method == "mice":
        imputer = IterativeImputer(max_iter=10, random_state=0)
    else:
        raise ValueError(f"Method '{method}' not supported for dataset-wide imputation.")
    
    # Fit and transform the entire DataFrame.
    imputed_array = imputer.fit_transform(impute_df)
    imputed_df = pd.DataFrame(imputed_array, columns=impute_df.columns, index=impute_df.index)
    
    # For each originally categorical column, round and decode back to strings.
    for col in categorical_cols:
        le = label_encoders[col]
        imputed_df[col] = imputed_df[col].round().astype(int)
        imputed_df[col] = le.inverse_transform(imputed_df[col])
    
    return imputed_df

def update_dataset_with_imputation(col, method, constant_value):
    # For constant imputation, format key as "constant (value)"
    key = method if method != "constant" else f"constant ({constant_value})"
    if method != "complete-case":
        imputed_series = data_store["imputed_versions"][col][key]
        data_store["original_data"][col] = imputed_series
    missing_matrix = data_store["original_data"].isna().astype(int).values.tolist()
    return missing_matrix

# -------------------------------------------------
# Route: Push a Chosen Imputed Column to Main DataFrame
# -------------------------------------------------
@app.route("/push-imputation", methods=["POST"], endpoint="push_imputation_route")
def push_imputation_route():
    try:
        req = request.json
        col = req.get("column")
        method = req.get("method")
        constant_value = req.get("constant_value")  # For constant imputation
        if not col or not method:
            return jsonify({"error": "Both column and method must be specified."}), 400

        if method in ["knn", "mice"]:
            # For KNN/MICE, impute the entire dataset.
            imputed_df = impute_with_sklearn_dataset_with_labelencoder(data_store["original_data"], method)
            data_store["original_data"].update(imputed_df)
            missing_matrix = data_store["original_data"].isna().astype(int).values.tolist()
            return jsonify({
                "message": f"Dataset imputed using '{method}' method.",
                "updated_matrix": missing_matrix
            })
        else:
            # For constant and column-specific methods.
            if method == "constant":
                if constant_value is None or str(constant_value).strip() == "":
                    return jsonify({"error": "A constant value must be provided for constant imputation."}), 400
                matrix = update_dataset_with_imputation(col, method, constant_value)
            else:
                matrix = update_dataset_with_imputation(col, method, None)
            return jsonify({
                "message": f"Imputation '{method}' for column '{col}' pushed to main dataset.",
                "updated_matrix": matrix
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# Route: Impute Entire Dataset
# -------------------------------------------------
@app.route("/impute-dataset", methods=["POST"])
def impute_dataset():
    try:
        req = request.json
        method = req.get("method")
        cval = req.get("constant_value")
        if not method:
            return jsonify({"error": "No imputation method provided."}), 400
        df = data_store["original_data"].copy()
        if method in ["knn", "mice"]:
            # Use the dataset-wide imputation with label encoding.
            imputed_df = impute_with_sklearn_dataset_with_labelencoder(df, method)
            data_store["original_data"] = imputed_df.copy()
        else:
            # Otherwise, process each column individually.
            for col in df.columns:
                if df[col].isna().sum() > 0:
                    try:
                        imputed_col = apply_imputation(col, method, cval)
                        df[col] = imputed_col
                    except Exception as ex:
                        print(f"Skipping imputation for column {col} due to error: {ex}")
            data_store["original_data"] = df.copy()
        matrix = data_store["original_data"].isna().astype(int).values.tolist()
        return jsonify({
            "message": f"Dataset imputed using method '{method}'.",
            "updated_matrix": matrix
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# Existing Routes: Uploading and Column Data
# -------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process-dataset", methods=["POST"])
def process_dataset_route():
    try:
        file = request.files.get("file")
        if not file or not file.filename.endswith(".csv"):
            return jsonify({"error": "Only CSV files are allowed."}), 400
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)
        df = pd.read_csv(path)
        os.remove(path)
        data_store["original_data"] = df
        data_store["imputed_versions"] = {}
        data_store["imputed_stats"] = {}
        data_store["pipeline_steps"] = []
        missing_matrix = df.isna().astype(int).values.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [col for col in df.columns if col not in numeric_cols]
        rows = df.index.tolist()
        return jsonify({
            "matrix": missing_matrix,
            "columns": df.columns.tolist(),
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "rows": rows
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/column-data", methods=["POST"])
def column_data():
    try:
        req = request.json
        col = req.get("column")
        if not col:
            return jsonify({"error": "No column provided."}), 400
        df = data_store["original_data"]
        if df is None:
            return jsonify({"error": "No dataset loaded."}), 400
        if col not in df.columns:
            return jsonify({"error": f"Column '{col}' not found in dataset."}), 400
        col_data = df[col]
        distribution_values = col_data.dropna().tolist()
        missing_values = col_data.isna().astype(int).tolist()
        kde_x, kde_y = [], []
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        if is_numeric:
            kde_x, kde_y = compute_kde(distribution_values)
        return jsonify({
            "distribution_values": distribution_values,
            "missing_values": missing_values,
            "kde_x": kde_x,
            "kde_y": kde_y,
            "is_numeric": is_numeric
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/impute", methods=["POST"])
def impute_route():
    try:
        data = request.json
        col = data.get("column")
        method = data.get("method")
        cval = data.get("constant_value")
        df = data_store["original_data"]
        if df is None:
            return jsonify({"error": "No dataset loaded."}), 400
        if col not in df.columns:
            return jsonify({"error": f"Column '{col}' not found."}), 400
        original_col = df[col].copy()
        if original_col.isna().sum() == 0 and method != "complete-case":
            return jsonify({"error": f"No missing values in column '{col}'."}), 400

        imputed_col = apply_imputation(col, method, cval)
        if col not in data_store["imputed_versions"]:
            data_store["imputed_versions"][col] = {}
        key = method if method != "constant" else f"{method} ({cval})"
        data_store["imputed_versions"][col][key] = imputed_col
        data_store["pipeline_steps"].append({
            "column": col,
            "method": method,
            "config": {"value": cval}
        })
        orig_vals = original_col.dropna().tolist()
        imp_vals = imputed_col.tolist()
        is_numeric = pd.api.types.is_numeric_dtype(original_col)
        if is_numeric:
            kde_x_orig, kde_y_orig = compute_kde(orig_vals)
            kde_x_imp, kde_y_imp = compute_kde(imp_vals)
            stats = compute_comparative_stats(orig_vals, imp_vals, (kde_x_orig, kde_y_orig), (kde_x_imp, kde_y_imp))
        else:
            stats = compute_categorical_stats(orig_vals, imp_vals)
        if method == "constant":
            stats["constant_value"] = cval
        if col not in data_store["imputed_stats"]:
            data_store["imputed_stats"][col] = {}
        data_store["imputed_stats"][col][key] = stats
        matrix = data_store["original_data"].isna().astype(int).values.tolist()
        return jsonify({
            "message": f"Imputation '{method}' applied to '{col}'.",
            "original_distribution": orig_vals,
            "imputed_distribution": imp_vals,
            "kde_x_original": kde_x_orig if is_numeric else [],
            "kde_y_original": kde_y_orig if is_numeric else [],
            "kde_x_imputed": kde_x_imp if is_numeric else [],
            "kde_y_imputed": kde_y_imp if is_numeric else [],
            "stats": stats,
            "updated_matrix": matrix
        })
    except Exception as e:
        return jsonify({"error": f"Imputation failed: {str(e)}"}), 500

@app.route('/get_models', methods=['GET'])
def get_models():
    try:
        model_data = str(ollama.list())
        pattern = r"model='(.*?)'"
        models = re.findall(pattern, model_data)
        models = [name.strip() for name in models if name.strip()]
        return jsonify({"success": True, "models": models})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/set_model', methods=['POST'])
def set_model():
    try:
        selected_model = request.form.get('model')
        if not selected_model:
            return jsonify({"success": False, "error": "No model selected."}), 400
        session['selected_model'] = selected_model
        return jsonify({"success": True, "message": f"Model '{selected_model}' set successfully."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/get-stats", methods=["POST"])
def get_stats_route():
    try:
        req = request.json
        col = req.get("column")
        col_stats = data_store["imputed_stats"].get(col, {})
        return jsonify({"stats": col_stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get-imputations", methods=["POST"])
def get_imputations():
    try:
        req = request.json
        col = req.get("column")
        col_imps = data_store["imputed_versions"].get(col, {})
        response = {m: s.tolist() for m, s in col_imps.items()}
        return jsonify({"imputations": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/save-pipeline", methods=["POST"])
def save_pipeline():
    try:
        pipeline_data = request.json.get("pipeline", [])
        if not isinstance(pipeline_data, list):
            return jsonify({"error": "Invalid pipeline format; must be a list."}), 400
        spath = os.path.join(PIPELINE_SAVE_DIR, "pipeline.json")
        with open(spath, "w") as f:
            json.dump(pipeline_data, f, indent=2)
        return jsonify({"success": True, "message": "Pipeline saved."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download-imputed-csv", methods=["GET"])
def download_imputed_csv():
    try:
        if data_store["original_data"] is None:
            return jsonify({"error": "No dataset loaded."}), 400
        df = data_store["original_data"].copy()
        csv_str = df.to_csv(index=False)
        response = make_response(csv_str)
        response.headers["Content-Disposition"] = "attachment; filename=imputed_dataset.csv"
        response.mimetype = "text/csv"
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------
# AI Summary Generation via Ollama
# -------------------------------------------------
@app.route("/generate_summary", methods=["POST"])
def generate_summary():
    global custom_prompt, default_prompt
    try:
        stats_table_html = request.json.get("stats_table_html")
        if not stats_table_html:
            return jsonify({"success": False, "error": "No imputation statistics table provided."}), 400

        soup = BeautifulSoup(stats_table_html, "html.parser")
        rows = soup.find_all("tr")
        table_text = ""
        for row in rows:
            cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
            table_text += " | ".join(cells) + "\n"

        current_prompt = custom_prompt or default_prompt
        prompt = current_prompt.format(table=table_text)

        selected_model = session.get("selected_model")
        if not selected_model:
            return jsonify({"success": False, "error": "No Ollama model selected. Please set a model first."}), 400

        response = ollama.chat(model=selected_model, messages=[{"role": "user", "content": prompt}])
        summary = response.get("message", {}).get("content")
        if not summary:
            return jsonify({"success": False, "error": "Received empty summary from Ollama."}), 500

        return jsonify({"success": True, "summary": summary})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/calculate-correlation", methods=["POST"])
def calculate_correlation():
    try:
        req = request.json
        column = req.get("column")
        imputed_values = req.get("imputed", None)
        if not column:
            return jsonify({"error": "No column specified."}), 400

        df = data_store["original_data"]
        if df is None:
            return jsonify({"error": "No dataset loaded."}), 400
        if column not in df.columns:
            return jsonify({"error": f"Column '{column}' not found in the dataset."}), 400

        numeric_df = df.select_dtypes(include=[np.number])
        if column not in numeric_df.columns:
            return jsonify({"error": f"Column '{column}' is not numeric."}), 400

        if imputed_values:
            numeric_df[column] = pd.Series(imputed_values)

        correlations = numeric_df.corr()[column].drop(index=column).to_dict()
        response = {
            "columns": list(correlations.keys()),
            "correlations": list(correlations.values()),
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------
# Run the App
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
