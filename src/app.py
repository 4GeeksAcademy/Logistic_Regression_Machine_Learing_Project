from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_one_hot_encoder() -> OneHotEncoder:
	try:
		return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
	except TypeError:
		return OneHotEncoder(handle_unknown="ignore", sparse=True)


def main() -> None:
	project_root = Path(__file__).resolve().parent.parent
	data_path = project_root / "data" / "raw" / "bank-marketing-campaign-data.csv"
	model_path = project_root / "models" / "logistic_regression_pipeline.joblib"

	if not data_path.exists():
		raise FileNotFoundError(f"Dataset not found at: {data_path}")

	df = pd.read_csv(data_path, sep=";")
	X = df.drop(columns=["y"])
	y = df["y"].map({"no": 0, "yes": 1})

	X_train_raw, X_test_raw, y_train, y_test = train_test_split(
		X, y, test_size=0.2, stratify=y, random_state=42
	)

	categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
	preprocess = ColumnTransformer(
		transformers=[("cat", build_one_hot_encoder(), categorical_cols)],
		remainder="passthrough",
	)

	model = Pipeline(
		steps=[
			("preprocess", preprocess),
			(
				"clf",
				LogisticRegression(
					solver="liblinear",
					C=1.0,
					max_iter=200,
					random_state=42,
				),
			),
		]
	)

	model.fit(X_train_raw, y_train)
	baseline_accuracy = accuracy_score(y_test, model.predict(X_test_raw))

	param_distributions = {
		"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
		"clf__solver": ["liblinear"],
		"clf__penalty": ["l2"],
		"clf__class_weight": [None, "balanced"],
		"clf__max_iter": [300, 600],
	}

	random_search = RandomizedSearchCV(
		estimator=model,
		param_distributions=param_distributions,
		n_iter=8,
		cv=3,
		scoring="accuracy",
		n_jobs=-1,
		random_state=42,
	)
	random_search.fit(X_train_raw, y_train)

	best_model = random_search.best_estimator_
	tuned_accuracy = best_model.score(X_test_raw, y_test)

	model_path.parent.mkdir(parents=True, exist_ok=True)
	dump(best_model, model_path)

	print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
	print(f"Baseline accuracy: {baseline_accuracy:.4f}")
	print(f"Best CV accuracy: {random_search.best_score_:.4f}")
	print(f"Tuned test accuracy: {tuned_accuracy:.4f}")
	print(f"Best params: {random_search.best_params_}")
	print(f"Saved model to: {model_path}")


if __name__ == "__main__":
	main()
