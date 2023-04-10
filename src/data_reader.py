from pathlib import Path
import json
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, rootdir):
        self.rootdir = Path(rootdir)
        self.img_filenames = sorted(list(self.rootdir.glob("*.jpg")))

    def __getitem__(self, idx):
        filename = self.img_filenames[idx]
        image = Image.open(self.img_filenames[idx])
        with open(str(filename) + ".json", "r") as f:
            label = json.load(f)

        return image, label

    def __len__(self):
        return len(self.img_filenames)
