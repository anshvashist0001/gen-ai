"""Compare BCE-only and BCE plus feature-matching generator objectives on toy data."""
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train(epochs=50,batch_size=128,seed=42,feature_weight=0.0):
    torch.manual_seed(seed)
    latent = 64
    generator = nn.Sequential(nn.Linear(latent,256),nn.ReLU(),nn.Linear(256,784),nn.Tanh())
    discriminator = nn.Sequential(nn.Linear(784,256),nn.LeakyReLU(.2),nn.Linear(256,1))
    g_opt = torch.optim.Adam(generator.parameters(),lr=.0002)
    d_opt = torch.optim.Adam(discriminator.parameters(),lr=.0002)
    bce, mse = nn.BCEWithLogitsLoss(),nn.MSELoss()
    initial = [p.detach().clone() for p in generator.parameters()]
    history = []
    for _ in range(epochs):
        # Bounded toy vectors match the output range of tanh. These are not images.
        real = torch.rand(batch_size,784)*2-1
        fake = generator(torch.randn(batch_size,latent))
        d_loss = bce(discriminator(real),torch.ones(batch_size,1)) + bce(discriminator(fake.detach()),torch.zeros(batch_size,1))
        d_opt.zero_grad(); d_loss.backward(); d_opt.step()
        for p in discriminator.parameters():
            p.requires_grad_(False)
        fake = generator(torch.randn(batch_size,latent))
        adversarial = bce(discriminator(fake),torch.ones(batch_size,1))
        real_features = discriminator[1](discriminator[0](real)).detach()
        fake_features = discriminator[1](discriminator[0](fake))
        matching = mse(fake_features.mean(0),real_features.mean(0))
        loss = adversarial + feature_weight*matching
        g_opt.zero_grad(); loss.backward(); g_opt.step()
        for p in discriminator.parameters():
            p.requires_grad_(True)
        history.append(float(loss.detach()))
    changed = any(not torch.equal(before,after.detach()) for before,after in zip(initial,generator.parameters()))
    return history,changed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--batch-size',type=int,default=128)
    args = parser.parse_args()
    if args.epochs<1 or args.batch_size<1:
        parser.error('epochs and batch-size must be positive')
    for weight,label in [(0.0,'BCE only'),(1.0,'BCE + feature matching')]:
        losses,updated = train(args.epochs,args.batch_size,feature_weight=weight)
        assert updated,'Generator parameters did not update'
        print(f'{label}: final objective={losses[-1]:.4f}; generator updated={updated}')
        plt.plot(losses,label=label)
    plt.title('Toy GAN objectives (different scales; not a quality ranking)')
    plt.xlabel('Update step'); plt.ylabel('Generator objective'); plt.legend(); plt.show()
