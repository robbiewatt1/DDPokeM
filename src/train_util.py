import torch
import torchvision
from UNet import UNet
from Dataset import PokemonDataset
from Diffusion import Diffusion
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_step(model, diffusion, data_loader, optimiser, step,
                    writer=None):
    """
    Function for a single training step.
    :param model: instance of the Unet class
    :param diffusion: instance of the Diffusion class
    :param data_loader: data loader
    :param optimiser: optimiser to use
    :param step: current step
    :param writer: tensorboard writer
    :return: loss value
    """

    model.train()

    epoch_losses = []
    with tqdm(total=len(data_loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {step}")
        for i, image in enumerate(data_loader):
            tq.update(1)

            image = image.to(diffusion.device)
            time = torch.randint(0, diffusion.steps, (image.shape[0],),
                                 device=diffusion.device)
            loss = diffusion.training_loss(model, image, time)
            loss.backward()

            if (i + 1) % 2 == 0:
                optimiser.step()
                optimiser.zero_grad()

                if writer is not None:
                    writer.add_scalar("Loss/train", loss.item(),
                                      step * len(data_loader) + i)

            epoch_losses.append(loss.item())
            tq.set_postfix_str(s=f"Loss: {loss.item():.4f}")

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        tq.set_postfix_str(s=f"Loss: {mean_loss:.4f}")
    return mean_loss


def sample_step(model, diffusion, n_sample):
    """
    Function for a sampling from the diffusion model.
    :param model: instance of the Unet class
    :param diffusion: instance of the Diffusion class
    :param n_sample: number of samples to generate
    :param writer: tensorboard writer
    :return: fig, ax
    """
    model.eval()

    images = diffusion.p_sample_full(model, n_sample)
    images = (PokemonDataset.inverse_transform(images) / 255.
              )
    fig, ax = plt.subplots(figsize=(20, 8), facecolor='white')
    grid_img = torchvision.utils.make_grid(
        images, nrow=images.shape[0]//2, padding=True, pad_value=1)
    ax.imshow(grid_img.permute(1, 2, 0).cpu().detach().numpy())
    ax.axis('off')
    return fig, ax

def train_generator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 64
    learning_rate = 1e-4
    num_epochs = 10000
    num_diffusion_timesteps = 4000
    unet_model = UNet(output_channels=6, num_res_blocks=2)
    unet_model.to(device)

    dataset = PokemonDataset()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=4)
    optimiser = torch.optim.Adam(unet_model.parameters(), lr=learning_rate)
    writer = SummaryWriter("./runs")

    diffusion = Diffusion(num_diffusion_timesteps, (3, 64, 64),
                          device=device, betas_method="cosine",
                          var_method="learned", loss_method="both")

    for epoch in range(num_epochs):
        train_step(unet_model, diffusion, dataloader, optimiser, epoch,
                   writer=writer)

        if epoch % 100 == 0:
            fig, ax = sample_step(unet_model, diffusion, 16)
            fig.savefig(f"./results/{epoch}.png", dpi=300)
            plt.close(fig)

if __name__ == "__main__":
    train_generator()