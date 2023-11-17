import torch
import torchvision
from datasets import load_dataset


class PokemonDataset(torch.utils.data.Dataset):
    """
    Dataset class of Pokemon images. Uses huggingface's datasets library from
    source https://huggingface.co/datasets/huggan/pokemon.
    """

    def __init__(self, image_shape=(32, 32)):
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


class PokemonUpscaleDataset(torch.utils.data.Dataset):
    def __init__(self, input_shape=(32, 32), output_shape=(128, 128)):
        super().__init__()
        """
        :param input_shape: shape of the input image
        :param output_shape: shape of the output image
        """
        dataset = load_dataset('huggan/pokemon', split='train')
        self.data = dataset['image']
        to_tensor = torchvision.transforms.ToTensor()
        for i, image in enumerate(self.data):
            if (to_tensor(image).shape[1] < 128
                    or to_tensor(image).shape[2] < 128):
                self.data.pop(i)

        self.base_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda t: t.float()),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.Lambda(lambda t: (t * 2.) - 1.),
        ])

        self.input_resize = torchvision.transforms.Resize(
            input_shape,
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            antialias=True)

        self.output_resize = torchvision.transforms.Resize(
            output_shape,
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            antialias=True)

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
        image = self.base_transform(self.data[index])
        return self.input_resize(image), self.output_resize(image)

    @staticmethod
    def inverse_transform(image):
        """Convert tensors from [-1., 1.] to [0., 255.]"""
        return ((image.clamp(-1, 1) + 1.0) / 2.0) * 255.0
