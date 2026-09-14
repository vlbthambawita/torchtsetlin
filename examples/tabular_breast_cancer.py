"""Interpretable rules on the Wisconsin breast cancer dataset (scikit-learn required)."""

import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import torchtsetlin as tt


def main() -> None:
    tt.seed_everything(0)
    data = load_breast_cancer()
    x_tr, x_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.3, random_state=0)
    x_tr = torch.tensor(x_tr, dtype=torch.float32)
    x_te = torch.tensor(x_te, dtype=torch.float32)

    enc = tt.data.ThermometerEncoder(n_bits=8, strategy="quantile").fit(x_tr)
    xb_tr, xb_te = enc(x_tr), enc(x_te)

    model = tt.TsetlinMachine(xb_tr.shape[1], 2, n_clauses=100, T=20, s=4.0, max_included_literals=6)
    model.feature_names = enc.feature_names(list(data.feature_names))
    model.class_names = list(data.target_names)

    trainer = tt.Trainer(model, batch_size=8, callbacks=[tt.EarlyStopping(patience=30)])
    trainer.fit((xb_tr, y_tr), epochs=200, val_data=(xb_te, y_te))
    print(trainer.test((xb_te, y_te)))

    print("\nRules voting for a class:")
    for rule in model.rules(polarity=1)[:8]:
        print(" ", rule)

    imp = tt.interpret.global_feature_importance(model, class_index=0)
    per_feature = tt.interpret.aggregate_literal_importance(imp, [i // 8 for i in range(xb_tr.shape[1])])
    order = per_feature.argsort(descending=True)[:10]
    print("\nMost used features for class", model.class_names[0])
    for i in order:
        print(f"  {data.feature_names[i]:25s} {per_feature[i]:.3f}")


if __name__ == "__main__":
    main()
