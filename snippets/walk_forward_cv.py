"""
Walk-forward (expanding-window) cross-validation for the trading model.

The single most important discipline in time-series ML: never let the model see
the future. A normal random k-fold split would train on tomorrow to predict today
and report fantasy accuracy. Here each fold trains only on data strictly *before*
its test window, and the training window grows forward through time — exactly how
the model would actually be used in production.

Excerpt from the ML-gated trading system. See docs/trading-system.md.
"""
import pandas as pd


def walk_forward_eval(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """Time-series aware walk-forward cross-validation."""
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    import lightgbm as lgb

    fold_size = len(X) // (n_splits + 1)
    metrics = []

    for i in range(n_splits):
        # Expanding train window; test on the *next* contiguous block only
        train_end = (i + 1) * fold_size
        test_end = train_end + fold_size
        if test_end > len(X):
            break

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

        # Skip degenerate folds (one class only) — metrics would be meaningless
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            class_weight="balanced",   # trade signals are heavily imbalanced
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.55).astype(int)   # gate above 0.5: precision over recall

        metrics.append({
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "auc": roc_auc_score(y_test, probs),
            "n_train": len(X_train),
            "n_test": len(X_test),
        })

    # Average across folds — one honest, leak-free performance estimate
    if not metrics:
        return {}
    keys = ("precision", "recall", "f1", "auc")
    return {k: sum(m[k] for m in metrics) / len(metrics) for k in keys}
