from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class VoiceCommandsDataset(Dataset):
    def __init__(self, df, audio_dir, max_samples=16000):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.max_samples = max_samples

    def __len__(self):
        return len(self.df)

    def _normalize_length(self, audio):
        if audio.shape[0] > self.max_samples:
            return audio[:self.max_samples]
        if audio.shape[0] < self.max_samples:
            padding = self.max_samples - audio.shape[0]
            return np.pad(audio, (0, padding), mode='constant')
        return audio

    def _load_audio(self, file_name):
        path = self.audio_dir / file_name
        arr = np.load(path, allow_pickle=False)
        if isinstance(arr, np.lib.npyio.NpzFile):
            key = 'audio' if 'audio' in arr.files else arr.files[0]
            arr = arr[key]
        return np.asarray(arr).reshape(-1).astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio = self._load_audio(row['filename'])
        audio = self._normalize_length(audio)
        label = int(row['label_id'])
        return torch.tensor(audio, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def load_data(train_metadata_csv, train_audio_dir, val_size=0.2, seed=42, batch_size=64, num_workers=0):
    train_df = pd.read_csv(train_metadata_csv)
    train_df = train_df.dropna(subset=['label']).copy()
    train_df['label'] = train_df['label'].astype(str)

    label_encoder = LabelEncoder()
    train_df['label_id'] = label_encoder.fit_transform(train_df['label'])
    num_classes = len(label_encoder.classes_)

    train_split_df, val_split_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed,
        stratify=train_df['label_id'],
    )

    train_dataset = VoiceCommandsDataset(train_split_df, audio_dir=train_audio_dir)
    val_dataset = VoiceCommandsDataset(val_split_df, audio_dir=train_audio_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, label_encoder, num_classes


def load_adv_test(test_dir, label_encoder, batch_size=64):
    from pathlib import Path as P
    test_dir = P(test_dir)
    adv_meta_path = test_dir / 'test_metadata.csv'
    if not adv_meta_path.exists():
        return None, 0

    adv_meta = pd.read_csv(adv_meta_path)
    if 'label' not in adv_meta.columns:
        return None, 0

    adv_meta['label'] = adv_meta['label'].astype(str)
    adv_meta['label_id'] = label_encoder.transform(adv_meta['label'])
    adv_dataset = VoiceCommandsDataset(adv_meta, test_dir)
    adv_loader = DataLoader(
        adv_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    return adv_loader, len(adv_dataset)
