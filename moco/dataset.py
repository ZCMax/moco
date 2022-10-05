from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader, has_file_allowed_extension
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import os
import random

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def find_instances(directory: str) -> Tuple[List[str]]:
    """Finds the instances folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    instances = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not instances:
        raise FileNotFoundError(f"Couldn't find any instances folder in {directory}.")
    samples = []
    for instance in instances:
        instance_dir = os.path.join(directory, instance)
        samples.append(instance_dir)

    return samples

class MVImageDataset(VisionDataset):
    """A generic data loader where the multi-view instance-aware images are 
    arranged in this way by default: ::

        root/instance1/xxx.png
        root/insatance1/xxy.png
        root/instance1/[...]/xxz.png

        root/instance2/123.png
        root/instance2/nsdf3.png
        root/instance2/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        instance_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
    ) -> None:
        super(MVImageDataset, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        samples = self.find_instances(self.root)

        self.loader = loader
        self.samples = samples

    def find_instances(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the instance folders in a dataset structured as follows::

            directory/
            ├── instance_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── instance_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_instances(directory)

    def random_choice(self, instance_dir: str) -> Tuple[List[str], Dict[str, int]]:

        mv_imgs = []
        for root, _, fnames in sorted(os.walk(instance_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                mv_imgs.append(path)
        imgs = random.choices(mv_imgs, k=2)
        return imgs
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        instance_dir = self.samples[index]
        imgs = self.random_choice(instance_dir)
        sample = [self.loader(img) for img in imgs]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        return len(self.samples)
