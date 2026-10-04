import torch
from experiment10 import train


def test_both_generator_objectives_update_parameters():
    torch.set_num_threads(1)
    for weight in [0.0,1.0]:
        losses,updated = train(epochs=2,batch_size=8,feature_weight=weight)
        assert updated
        assert all(torch.isfinite(torch.tensor(losses)))
