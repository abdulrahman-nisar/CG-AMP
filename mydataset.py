import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Batch  # optional (only used by collate_fn)
except Exception:  # pragma: no cover
    Batch = None


class MyDataset(Dataset):

    def __init__(self, data):
        self.feature1 = data[0]
        self.feature2 = data[1]
        self.labels = data[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(self.feature1[index]), torch.tensor(self.feature2[index]), self.labels[index]


def _pad_esm_to_len(esm_tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    # Accept (L, 1280) or (1, L, 1280) and return (target_len, 1280).
    if esm_tensor.dim() == 3 and esm_tensor.size(0) == 1:
        esm_tensor = esm_tensor.squeeze(0)
    if esm_tensor.dim() != 2:
        raise ValueError(f"Unexpected ESM tensor shape: {tuple(esm_tensor.shape)}")

    length, dim = esm_tensor.shape
    if length == target_len:
        return esm_tensor
    if length > target_len:
        return esm_tensor[:target_len]

    pad = torch.zeros((target_len - length, dim), dtype=esm_tensor.dtype)
    return torch.cat([esm_tensor, pad], dim=0)

def collate_fn(batch):
    if Batch is None:
        raise ImportError(
            "torch_geometric is not installed, but collate_fn() requires it. "
            "Install torch_geometric or use collate_fn1/collate_fn2 instead."
        )
    data, labels = map(list, zip(*batch))
    if len(batch) % 2 != 0:
        data = data[:-1]
        labels = labels[:-1]
    # data = torch.tensor(data)
    labels = torch.tensor(labels)
    device = torch.device("cuda")
    data1_ls = []
    data2_ls = []
    label1_ls = []
    label2_ls = []
    label_ls = []
    batch_size = len(batch)
    mid = batch_size // 2
    for i in range(mid):
        data1, label1 = data[i], labels[i]
        data2, label2 = data[i + int(batch_size / 2)], labels[i + int(batch_size / 2)]
        label1_ls.append(label1.unsqueeze(0))
        label2_ls.append(label2.unsqueeze(0))
        label = (label1 ^ label2)
        data1_ls.append(data1)
        data2_ls.append(data2)
        label_ls.append(label.unsqueeze(0))
    data1 = data[:mid]
    data2 = data[mid:]
    label = torch.cat(label_ls)
    label1 = torch.cat(label1_ls)
    label2 = torch.cat(label2_ls)
    return Batch.from_data_list(data), Batch.from_data_list(data1), Batch.from_data_list(data2), label, label1, label2

def collate_fn1(batch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    esm, data, labels = zip(*batch)
    if len(batch) % 2 != 0:
        esm = esm[:-1]
        data = data[:-1]
        labels = labels[:-1]

    # data tensors are already padded to the global max_len from FASTA.
    # Pad ESM embeddings to the same length so we can stack.
    target_len = int(data[0].shape[0])
    esm = tuple(_pad_esm_to_len(t, target_len) for t in esm)

    batch_size = len(batch) // 2
    esm1_ls = esm[:batch_size]
    esm2_ls = esm[batch_size:]
    data1_ls = data[:batch_size]
    data2_ls = data[batch_size:]
    labels1_ls = labels[:batch_size]
    labels2_ls = labels[batch_size:]
    # Always use float32 for model compatibility (embeddings may be saved as float16).
    esm = torch.stack(esm).to(device).float()
    esm1 = torch.stack(tuple(_pad_esm_to_len(t, target_len) for t in esm1_ls)).to(device).float()
    esm2 = torch.stack(tuple(_pad_esm_to_len(t, target_len) for t in esm2_ls)).to(device).float()
    data = torch.stack(data).to(device)
    data1 = torch.stack(data1_ls).to(device)
    data2 = torch.stack(data2_ls).to(device)

    label_ = []
    for i in range(batch_size):
        label_.append(labels1_ls[i] ^ labels2_ls[i])
    labels = torch.tensor(labels).to(device, dtype=torch.float32)
    label1 = torch.tensor(labels1_ls).to(device, dtype=torch.float32)
    label2 = torch.tensor(labels2_ls).to(device, dtype=torch.float32)
    label_ = torch.tensor(label_).to(device, dtype=torch.float32)

    esm_dic = {'esm': esm, 'esm1': esm1, 'esm2': esm2}
    data_dic = {'data': data, 'data1': data1, 'data2': data2}
    label_dic = {'labels': labels, 'label1': label1, 'label2': label2, 'label_': label_}
    return esm_dic, data_dic, label_dic

def collate_fn2(batch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    esm, data, labels = map(list, zip(*batch))
    target_len = int(data[0].shape[0])
    # Always use float32 for model compatibility (embeddings may be saved as float16).
    esm = torch.stack([_pad_esm_to_len(t, target_len) for t in esm]).to(device).float()
    data = torch.stack(data).to(device)
    labels = torch.tensor(labels).to(device, dtype=torch.float32)
    return esm, data, labels.unsqueeze(1)





