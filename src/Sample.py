import torch
import torchvision
from UNet import UNet
from Dataset import PokemonDataset
from Diffusion import Diffusion
import matplotlib.pyplot as plt


def sample():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_diffusion_timesteps = 4000
    image_shape = (3, 128, 128)
    n_samples = 20

    diffusion_gen = Diffusion(num_diffusion_timesteps, image_shape,
                          device=device, betas_method="cosine",
                          var_method="learned", loss_method="both")

    gen_model = UNet(output_channels=6, num_res_blocks=3, base_channels=64)
    gen_model.load_state_dict(torch.load("./Models/model.pt"))
    gen_model.to(device)

    images = diffusion_gen.p_sample_full(gen_model, n_samples)
    images = (PokemonDataset.inverse_transform(images) / 255.)

    fig, ax = plt.subplots(figsize=(20, 8), facecolor='white')
    grid_img = torchvision.utils.make_grid(
        images, nrow=5, padding=True, pad_value=1)
    ax.imshow(grid_img.permute(1, 2, 0).cpu().detach().numpy())
    ax.axis('off')
    plt.show()

if __name__ == "__main__":
    sample()