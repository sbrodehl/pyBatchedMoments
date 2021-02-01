import multiprocessing
import tempfile

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from batchedmoments import BatchedMoments

bm = BatchedMoments(axis=(0, 2, 3))

with tempfile.TemporaryDirectory() as root:
    imagenet_data = datasets.FashionMNIST(
        str(root),
        download=True,
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    data_loader = DataLoader(
        imagenet_data,
        batch_size=1024,
        num_workers=multiprocessing.cpu_count()
    )
    for batch in data_loader:
        imgs, _ = batch
        bm(imgs)

print(f"Fashion-MNIST: mean={bm.mean} std={bm.std}")
