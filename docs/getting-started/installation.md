# Installation

`torchtsetlin` needs Python 3.9+ and PyTorch 2.0+. Install PyTorch first (pick the CUDA or
CPU build that matches your machine from [pytorch.org](https://pytorch.org/get-started/locally/)),
then:

```bash
pip install torchtsetlin
```

Optional extras:

```bash
pip install "torchtsetlin[viz]"      # matplotlib for torchtsetlin.viz
pip install "torchtsetlin[vision]"   # torchvision for the MNIST helpers
pip install "torchtsetlin[sklearn]"  # scikit-learn interop in the examples
pip install "torchtsetlin[all]"      # everything, including docs and dev tools
```

From source:

```bash
git clone https://github.com/vlbthambawita/torchtsetlin
cd torchtsetlin
pip install -e ".[dev]"
pytest            # run the test-suite (uses the GPU when available)
mkdocs serve      # browse this documentation locally
```

## Verify the installation

```python
import torch, torchtsetlin as tt
print(tt.__version__, torch.cuda.is_available())
model = tt.TsetlinMachine(n_features=4, n_classes=2, n_clauses=4, T=2)
print(model)
```
