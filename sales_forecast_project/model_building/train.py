import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow
import os

mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("Sales-Forecast-Superkart-Experiment")

# Hugging Face API authentication
api = HfApi(token=os.getenv("HF_TOKEN"))
Xtrain_path = "hf://datasets/MattVarg/sales-forecast-superkart/Xtrain.csv"
Xtest_path = "hf://datasets/MattVarg/sales-forecast-superkart/Xtest.csv"
ytrain_path = "hf://datasets/MattVarg/sales-forecast-superkart/ytrain.csv"
ytest_path = "hf://datasets/MattVarg/sales-forecast-superkart/ytest.csv"

# Load datasets
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze() # Use squeeze() for single column target
ytest = pd.read_csv(ytest_path).squeeze() # Use squeeze() for single column target

numeric_features = [
    'Product_Weight',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Establishment_Year',
    'Store_Location_City_Type'
]
categorical_features = [
    'Product_Sugar_Content',
    'Product_Type',
    'Store_Size',
    'Store_Type'
]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)
# Define base XGBoost model for regression
xgb_model = xgb.XGBRegressor(random_state=42)

# Define hyperparameter grid for XGBRegressor
param_grid = {
    'xgbregressor__n_estimators': [50, 100, 150],
    'xgbregressor__max_depth': [3, 5, 7],
    'xgbregressor__learning_rate': [0.01, 0.1, 0.2],
    'xgbregressor__subsample': [0.6, 0.8, 1.0],
    'xgbregressor__colsample_bytree': [0.6, 0.8, 1.0]
}
# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error') # Using a regression scoring metric
    grid_search.fit(Xtrain, ytrain)

    # Log hyperparameters
    mlflow.log_params(grid_search.best_params_)

    # Store the best model
    best_model = grid_search.best_estimator_

    # Make predictions on the training and test data
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Evaluation with regression metrics
    train_mae = mean_absolute_error(ytrain, y_pred_train)
    train_mse = mean_squared_error(ytrain, y_pred_train)
    train_rmse = mean_squared_error(ytrain, y_pred_train, squared=False)
    train_r2 = r2_score(ytrain, y_pred_train)

    test_mae = mean_absolute_error(ytest, y_pred_test)
    test_mse = mean_squared_error(ytest, y_pred_test)
    test_rmse = mean_squared_error(ytest, y_pred_test, squared=False)
    test_r2 = r2_score(ytest, y_pred_test)

    # Log metrics
    mlflow.log_metrics({
        "train_mae": train_mae,
        "train_mse": train_mse,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "test_mae": test_mae,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_r2": test_r2
    })

    # Save the model locally
    model_path = "sales_forecast_package_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "MattVarg/sales-forecast-package-model"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj="sales_forecast_package_model_v1.joblib",
        path_in_repo="sales_forecast_package_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
