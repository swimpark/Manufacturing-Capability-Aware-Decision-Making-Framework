import os
import torch
from torch.utils.data import Dataset
import binvox_io
import pandas as pd
import numpy as np

# +

class VoxelDataset(Dataset):
    def __init__(self, csv_file, voxel_dir, use_cache=False, source="binvox"):
        self.df = pd.read_csv(csv_file)
        if "filename" not in self.df.columns:
            if "FileName" in self.df.columns:
                self.df.rename(columns={"FileName": "filename"}, inplace=True)
            else:
                raise KeyError("CSV must contain 'filename' or 'FileName'.")

        self.voxel_dir = voxel_dir
        self.source = source

        if self.source == "pt":
            pt_dir = os.path.join(self.voxel_dir, "pt_cache")
            self.voxel_files = [
                os.path.join(
                    pt_dir,
                    os.path.splitext(os.path.basename(str(fname)))[0] + ".pt",
                )
                for fname in self.df["filename"]
            ]
        elif self.source == "binvox":
            self.voxel_files = [
                os.path.join(self.voxel_dir, str(fname))
                for fname in self.df["filename"]
            ]
        else:
            raise ValueError("source must be either 'pt' or 'binvox'.")

        missing = [path for path in self.voxel_files if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} voxel files. First missing file: {missing[0]}"
            )

        self.use_cache = use_cache

        if self.use_cache:
            self._cache = [None] * len(self.df)
        else:
            self._cache = None

        print(f"[VoxelDataset] CSV rows = {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def _load_voxel_from_file(self, idx):
        voxel_path = self.voxel_files[idx]
        if self.source == "pt":
            try:
                voxel_tensor = torch.load(
                    voxel_path, map_location="cpu", weights_only=True
                )
            except TypeError:
                voxel_tensor = torch.load(voxel_path, map_location="cpu")

            if voxel_tensor.ndim == 3:
                voxel_tensor = voxel_tensor.unsqueeze(0)
            elif voxel_tensor.ndim == 5 and voxel_tensor.size(0) == 1:
                voxel_tensor = voxel_tensor.squeeze(0)
            elif voxel_tensor.ndim != 4:
                raise RuntimeError(
                    f"Unexpected cached voxel shape: {tuple(voxel_tensor.shape)}"
                )
            return voxel_tensor.float()

        with open(voxel_path, "rb") as f:
            vox = binvox_io.read_as_3d_array(f).data  # bool array
        vox = vox.astype(np.float32)
        return torch.from_numpy(vox).unsqueeze(0)  # [1, D, H, W]

    def __getitem__(self, idx):
        if self.use_cache:
            if self._cache[idx] is None:
                voxel_tensor = self._load_voxel_from_file(idx)
                self._cache[idx] = voxel_tensor
            return self._cache[idx]
        else:
            return self._load_voxel_from_file(idx)
