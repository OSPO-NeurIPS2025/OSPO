
import os, json
from torch.utils.data import Dataset, DataLoader

class BaseDataset(Dataset):
    def __init__(self, 
                 fpath,
                 s_idx=None, 
                 e_idx=None): 

        with open(fpath, 'r') as f:
            self.data = json.load(f)

        # If 'p_method' is not included in keys, add it manually.
        if self.data[0].get("perturbed_method", None) is None:
            for sample in self.data:
                if sample["sub_category"] in ["attribute1_color", "attribute1_texture", "attribute1_shape", "layout2"]:
                    sample["perturbed_method"] = ["replace", "replace", "replace"]
                elif sample["sub_category"] == "non-spatial":
                    sample["perturbed_method"] = ["replace", "drop", "replace"]
                elif sample["sub_category"] in ["complex", "attribute2", "layout1", "layout3"]:
                    sample["perturbed_method"] = ["replace", "swap", "drop"]
                else:
                    raise ValueError(f"Unknown sub_category: {sample['sub_category']}")

        if s_idx is not None and e_idx is not None:
            self.data = self.data[s_idx:e_idx]
        elif s_idx is not None:
            self.data = self.data[s_idx:]
        elif e_idx is not None:
            self.data = self.data[:e_idx]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

