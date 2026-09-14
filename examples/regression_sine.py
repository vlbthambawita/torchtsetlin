"""Regression Tsetlin machine on a noisy 1-D function with thermometer inputs."""

import math

import torch

import torchtsetlin as tt


def main() -> None:
    tt.seed_everything(0)
    x = torch.rand(4000, 1) * 2 * math.pi
    y = 0.5 * (torch.sin(x[:, 0]) + 1) + 0.05 * torch.randn(4000)
    enc = tt.data.ThermometerEncoder(n_bits=32, strategy="uniform", value_range=(0, 2 * math.pi))
    xb = enc(x)

    model = tt.RegressionTsetlinMachine(xb.shape[1], n_clauses=400, T=200, s=2.0, y_range=(0.0, 1.0))
    trainer = tt.Trainer(model, batch_size=16)
    trainer.fit((xb[:3000], y[:3000]), epochs=40, val_data=(xb[3000:], y[3000:]))
    print(trainer.test((xb[3000:], y[3000:])))
    print(model.rules()[:5])


if __name__ == "__main__":
    main()
