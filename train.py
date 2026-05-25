from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATASET_PATH = Path("data/iris.csv")
MODEL_PATH = Path("modelo.pkl")


def main():
    df = pd.read_csv(DATASET_PATH)

    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    target = "species"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(
        {
            "model": model,
            "features": features,
            "accuracy": accuracy,
            "classes": sorted(y.unique().tolist()),
        },
        MODEL_PATH,
    )

    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Acurácia: {accuracy:.2%}")
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
