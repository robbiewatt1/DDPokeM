import torch
import torchvision
from datasets import load_dataset


class PokemonDataset(torch.utils.data.Dataset):
    """
    Dataset class of Pokemon images. Uses huggingface's datasets library from
    source https://huggingface.co/datasets/huggan/pokemon.
    """

    def __init__(self, image_shape=(64, 64) ):
        super().__init__()
        """
        :param path: path to the dataset
        :param transform: transform to apply to the dataset
        """
        dataset = load_dataset('huggan/pokemon', split='train')
        self.data = dataset['image']

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda t: t.float()),
            torchvision.transforms.Resize(image_shape,
             interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
             antialias=True),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.Lambda(lambda t: (t * 2.) - 1.),
        ])

    def __len__(self):
        """
        :return: length of the dataset
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        :param index: index of the dataset
        :return: input data and time data
        """
        return self.transform(self.data[index])

    @staticmethod
    def inverse_transform(image):
        """Convert tensors from [-1., 1.] to [0., 255.]"""
        return ((image.clamp(-1, 1) + 1.0) / 2.0) * 255.0