# Generative modeling experiments

A collection of Python lab exercises covering probability distributions,
parameter estimation, classifiers and generative models. Each script is a
standalone experiment. This is coursework-scale code, rather than a packaged
training framework or a deployed application.

## Setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python experiment1.py
```

The default PyTorch installation is sufficient for CPU execution. MNIST-based
scripts download the dataset into `./data` on first use, so run them from the
repository root with an internet connection. Training time depends on your machine.

## Experiments

| Script | What it demonstrates | Input |
|---|---|---|
| `experiment1.py` | Sampling normal, uniform, exponential and binomial distributions | Synthetic samples |
| `experiment2.py` | Maximum-likelihood normal parameter estimates | Synthetic normal data |
| `experiment3.py` | Feed-forward digit classification | MNIST |
| `experiment4.py` | GAN training with an MLP generator/discriminator | MNIST |
| `experiment5.py` | Affine coupling layers and invertible sampling | Synthetic 2D mixture |
| `experiment6.py` | Accuracy, macro precision/recall/F1 and confusion matrix | Fixed example labels |
| `experiment7.py` | Train/validation/test split for an MLP classifier | MNIST |
| `experiment8.py` | Variational autoencoder and reparameterization | MNIST |
| `experiment9.py` | LSTM next-token training and autoregressive sampling | Random integer sequences |
| `experiment10.py` | BCE and feature-matching generator objectives | Bounded random vectors |
| `project_parameter_estimation.py` | Normal/exponential MLE and sample-size effects | Synthetic samples |

## Running and reading results

Run one script at a time, for example `python experiment8.py`. Scripts print
progress or metrics and most open Matplotlib figures. Close a figure to continue
when a plotting call blocks execution. For a headless run, set `MPLBACKEND=Agg`.

Experiment 10 supports a shorter smoke run:

```bash
python experiment10.py --epochs 2 --batch-size 8
python -m pip install pytest
python -m pytest test_gan.py -q
```

Both generator objectives now perform optimizer updates. The test checks this
explicitly. The curves have different objective scales and should not be read as
a ranking of generated-sample quality. This experiment does not implement WGAN.

## Limitations

- Most exercises use fixed hyperparameters and do not save checkpoints.
- Experiment 6's labels are illustrative; they are not predictions from a model.
- Experiment 9 trains on independent random tokens, which contain no meaningful
  next-token structure. It demonstrates the mechanics, not useful language modeling.
- Experiment 5 leaves the first coordinate unchanged in every coupling layer.
  Alternating masks or permutations would be needed for a more expressive flow.
- Loss plots alone do not establish generative quality. Several training plots
  record the last minibatch rather than an epoch-wide average.
- Scores from synthetic exercises should not be presented as real-world results.

If MNIST cannot download, check network access and the `data/` directory. If Torch
or torchvision fails to import, install compatible versions in a clean environment.
