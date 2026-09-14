# Hyper-parameters

| Parameter | Meaning | Typical values |
|---|---|---|
| `n_clauses` | clauses per class (flat/conv) or shared clauses (coalesced) | 20 (XOR) … 2000–8000 (MNIST) |
| `T` | vote margin: how many net votes a class should reach | ≈ 0.5–2 × clauses/class for flat, up to 10 × for convolutional |
| `s` | specificity: `1/s` forget probability | 2–4 for small/tabular, 5–15 for images, 2–8 for text |
| `n_states` | memory depth `N` per action | 50–128 (256 states); deeper = slower forgetting |
| `boost_true_positive` | deterministic memorisation of True literals | `True` |
| `weighted` | integer clause weights | `True` for images, often `False` for tiny problems |
| `max_included_literals` | clause size budget | 8–32 for concise rules |
| `drop_clause_p` | fraction of clauses dropped per batch/epoch | 0.25 (images) – 0.75 (text) |
| `focused_negative_sampling` | pick confusable negative classes | `True` for many classes |
| `feedback_mode` | `"batch"` or `"sequential"` | `"batch"` with batch size 8–64 |

Reference configurations from the literature:

| Task | clauses | T | s | notes |
|---|---|---|---|---|
| Noisy XOR (12 bits) | 20 | 15 | 3.9 | 100 states, ~200 epochs |
| MNIST, flat | 2000/class | 50 | 10 | `pixel > 0.3`, 256 states, 400 epochs |
| MNIST, conv 10×10 | 8000/class | 10 000 | 5 | weighted, adaptive Gaussian threshold |
| Fashion-MNIST, conv 10×10 | 8000/class | 10 000 | 10 | weighted |
| CIFAR-10, conv 3×3 colour thermometer | 2000/class | 1500 | 2.5 | composites |
| IMDb bag-of-words | 10 000 | 8000 | 2.0 | drop clause 0.75 |

Rules of thumb:

* Raise `T` together with the number of clauses; the ratio matters more than either value.
* If clauses stay empty or tiny, raise `s`; if they memorise noise, lower `s`.
* Use `HyperparameterSchedule` to anneal `s` or grow `drop_clause_p` over epochs.
* Track `model.state_summary()` (mean included literals, empty clauses) with
  `StateSummaryLogger`.
