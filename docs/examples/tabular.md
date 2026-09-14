# Tabular data with thermometers

Full script: `examples/tabular_breast_cancer.py` (requires scikit-learn).

```python
import torch, torchtsetlin as tt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
x_tr, x_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.3, random_state=0)

enc = tt.data.ThermometerEncoder(n_bits=8, strategy="quantile").fit(torch.tensor(x_tr, dtype=torch.float32))
xb_tr, xb_te = enc(torch.tensor(x_tr, dtype=torch.float32)), enc(torch.tensor(x_te, dtype=torch.float32))

model = tt.TsetlinMachine(xb_tr.shape[1], 2, n_clauses=100, T=20, s=4.0, max_included_literals=6)
model.feature_names = enc.feature_names(data.feature_names)
model.class_names = list(data.target_names)

trainer = tt.Trainer(model, batch_size=8, callbacks=[tt.EarlyStopping(patience=30)])
trainer.fit((xb_tr, y_tr), epochs=200, val_data=(xb_te, y_te))
print(trainer.test((xb_te, y_te)))
for rule in model.rules(polarity=1)[:5]:
    print(rule)          # IF worst radius>=16.8 AND NOT mean texture>=21.5 THEN malignant

imp = tt.interpret.global_feature_importance(model, class_index=0)
per_feature = tt.interpret.aggregate_literal_importance(imp, [i // 8 for i in range(xb_tr.shape[1])])
tt.viz.plot_feature_importance(per_feature, data.feature_names, top_k=10)
```

The clause size constraint keeps rules short enough to read; the thermometer thresholds make
each literal a clinically interpretable inequality.
