import torch
import torchvision
from UNet import UNet
from Dataset import PokemonDataset
from Diffusion import Diffusion
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_step(model, diffusion, data_loader, optimiser, step,
               writer=None, batch_multiplier=1):
    """
    Function for a single training step.
    :param model: instance of the Unet class
    :param diffusion: instance of the Diffusion class
    :param data_loader: data loader
    :param optimiser: optimiser to use
    :param step: current step
    :param writer: tensorboard writer
    :param batch_multiplier: number of batches to accumulate gradients over
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

            if (i + 1) % batch_multiplier == 0:
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
    images = PokemonDataset.inverse_transform(images) / 255.
    fig, ax = plt.subplots(figsize=(20, 8), facecolor='white')
    grid_img = torchvision.utils.make_grid(
        images, nrow=images.shape[0]//2, padding=True, pad_value=1)
    ax.imshow(grid_img.permute(1, 2, 0).cpu().detach().numpy())
    ax.axis('off')
    return fig, ax


def train_generator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 128
    learning_rate = 1e-4
    num_epochs = 10000
    num_diffusion_timesteps = 4000
    image_shape = (3, 32, 32)

    # I want this to now run on multiple GPUs
    unet_model = UNet(output_channels=6, num_res_blocks=4, base_channels=128)
    unet_model.to(device)

    dataset = PokemonDataset(image_shape=(image_shape[1], image_shape[2]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=8)
    optimiser = torch.optim.Adam(unet_model.parameters(), lr=learning_rate)
    writer = SummaryWriter("./runs")

    diffusion = Diffusion(num_diffusion_timesteps, image_shape,
                          device=device, betas_method="cosine",
                          var_method="learned", loss_method="both")

    losses = []
    for epoch in range(num_epochs):
        loss = train_step(unet_model, diffusion, dataloader, optimiser, epoch,
                   writer=writer)
        losses.append(loss)

        if epoch % 100 == 0:
            fig, ax = sample_step(unet_model, diffusion, 16)
            fig.savefig(f"./results/{epoch}.png", dpi=300)
            plt.close(fig)

        # save the model
        if losses[-1] == min(losses):
            torch.save(unet_model.state_dict(), f"./Models/best_model.pt")
        if (epoch + 1) % 1000 == 0:
            torch.save(unet_model.state_dict(), f"./Models/{epoch}.pt")


if __name__ == "__main__":
    train_generator()
