import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from batchedmoments import BatchedMoments

if __name__ == '__main__':
    batchsize_exp = range(17)
    results = []
    results_file = Path("naive-bm-comparison.csv")
    if not results_file.exists():
        with tempfile.TemporaryDirectory() as root:
            image_data = datasets.FashionMNIST(
                str(root),
                download=True,
                train=True,
                transform=transforms.Compose([
                    transforms.ToTensor()
                ])
            )
            for exp in batchsize_exp:
                bm = BatchedMoments(axis=(0, 2, 3))
                # compare to naive solution
                means = []
                stds = []
                data_loader = DataLoader(
                    image_data,
                    batch_size=2**exp
                )
                for imgs, _ in data_loader:
                    imgs = imgs.numpy()

                    bm(imgs)

                    means.append(np.mean(imgs))
                    stds.append(np.std(imgs))

                naive_mean = np.mean(means, keepdims=True)
                naive_std = np.mean(stds, keepdims=True)

                results.append((2**exp, bm.mean.item(), bm.std.item(), naive_mean.item(), naive_std.item()))
                print(f"Fashion-MNIST (batched):\tbs={2**exp:5d} mean={bm.mean.item()} std={bm.std.item()}")
                # naive solution
                print(f"Fashion-MNIST (naive):\t\tbs={2**exp:5d} mean={np.mean(means, keepdims=True).item()} std={np.mean(stds, keepdims=True).item()}")
                df = pd.DataFrame(results, columns=["Batchsize", "Mean (BatchedMoments)", "Std (BatchedMoments)", "Mean (Naive)", "Std (Naive)"])
                df.to_csv("naive-bm-comparison.csv")

    df = pd.read_csv(results_file)

    # plot the differences
    fig, axs = plt.subplots(2, 1, sharex="all", dpi=300)
    fig.suptitle('FashionMNIST Sample Statistics')
    axs[0].plot(df["Batchsize"], df["Mean (BatchedMoments)"], label="ours")
    axs[0].plot(df["Batchsize"], df["Mean (Naive)"], label="naive")
    axs[0].legend()
    axs[0].set_ylabel('Mean')
    axs[0].grid(True)

    axs[1].plot(df["Batchsize"], df["Std (BatchedMoments)"], label="ours")
    axs[1].plot(df["Batchsize"], df["Std (Naive)"], label="naive")
    axs[1].legend()
    axs[1].set_ylabel('Std')
    axs[1].set_xlabel('Batchsize')
    axs[1].grid(True)

    fig.tight_layout()
    plt.savefig("naive-bm-comparison.png", dpi=300)
    plt.show(dpi=300)
