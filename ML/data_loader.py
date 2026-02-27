import os
import sys
# Ensure project root is in path regardless of CWD
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Generator

# Define core constants
DATA_PATH = os.path.join(_PROJECT_ROOT, "data/features/master_dataset.csv")
SELECTED_DIR = os.path.join(_PROJECT_ROOT, "data/features/selected")

class SpatialDataLoader:
    """
    Unified Data Loader for ML models (RF, CNN).
    Handles data loading, standardization, missing value imputation,
    and Spatial Leave-One-Field-Out (LOFO) Cross-Validation.
    """
    def __init__(self, target: str, scale_features: bool = True):
        self.target = target
        self.scale_features = scale_features
        self.features_list = self._load_selected_features()
        
        # Internal state
        self.df = None
        self.X = None
        self.y = None
        self.fields = None
        self.scaler = None
        
        self._load_and_preprocess()

    def _load_selected_features(self) -> List[str]:
        """Loads the top-15 selected features for the specific target."""
        filepath = os.path.join(SELECTED_DIR, f"{self.target}_best_features.txt")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Selected features not found at {filepath}")
            
        with open(filepath, 'r') as f:
            features = [line.strip() for line in f if line.strip()]
            
        if len(features) == 0:
            raise ValueError(f"Feature list for {self.target} is empty.")
        return features

    def _load_and_preprocess(self):
        """Loads master dataset, filters NaNs for the target, and optionally scales X."""
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Master dataset not found at {DATA_PATH}")
            
        # load raw data
        raw_df = pd.read_csv(DATA_PATH, low_memory=False)
        
        if self.target not in raw_df.columns:
            raise ValueError(f"Target column '{self.target}' not found in dataset.")
            
        # 1. Drop rows where target is NaN
        df_valid = raw_df.dropna(subset=[self.target]).reset_index(drop=True)
        self.y = df_valid[self.target].values
        self.fields = df_valid['field_name'].values
        
        # 2. Extract X
        missing_cols = [col for col in self.features_list if col not in df_valid.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns in DB: {missing_cols}")
            
        X_df = df_valid[self.features_list]
        
        # 3. Store raw X (no global imputation or scaling)
        self.X = X_df.values
        self.df = df_valid

    def get_fold_data(self, train_idx: np.ndarray, test_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes a single fold's data without leakage.
        1. Impute using TRAIN median
        2. Scale using TRAIN mean/std
        """
        X_train = self.X[train_idx].copy()
        X_test = self.X[test_idx].copy()
        y_train = self.y[train_idx]
        y_test = self.y[test_idx]

        # Median imputation based on TRAIN only
        train_median = np.nanmedian(X_train, axis=0)
        # Handle case where entire column is NaN in train (rare but possible)
        train_median[np.isnan(train_median)] = 0.0 
        
        # Apply to both
        for col_idx in range(X_train.shape[1]):
            X_train[np.isnan(X_train[:, col_idx]), col_idx] = train_median[col_idx]
            X_test[np.isnan(X_test[:, col_idx]), col_idx] = train_median[col_idx]

        # Scaling based on TRAIN only
        if self.scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test
        
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns X, y, and corresponding field names."""
        return self.X, self.y, self.fields
        
    def get_feature_names(self) -> List[str]:
        """Returns the names of the features used."""
        return self.features_list
        
    def iter_lofo_cv(self) -> Generator[Tuple[np.ndarray, np.ndarray, str], None, None]:
        """
        Generates train/test indices using Spatial Leave-One-Field-Out (LOFO) strategy.
        It isolates entire fields so spatial autocorrelation does not leak from train to test.
        
        Yields:
            train_idx (np.ndarray): Indices for training
            test_idx (np.ndarray): Indices for testing
            test_field_name (str): The name of the field left out
        """
        unique_fields = np.unique(self.fields)
        
        if len(unique_fields) < 2:
            raise ValueError("Not enough distinct fields to perform LOFO CV.")
            
        for uf in unique_fields:
            test_mask = (self.fields == uf)
            train_mask = ~test_mask
            
            # Convert boolean masks to integer indices
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
                
            yield train_idx, test_idx, str(uf)

    def get_pytorch_dataloaders(self, patches_dir: str, patch_size: int, batch_size: int = 32, num_workers: int = 4):
        """
        Generates PyTorch DataLoaders using Spatial LOFO CV.
        Yields:
             train_loader: DataLoader for train patches
             test_loader: DataLoader for test patches
             test_field_name: Name of the left out field
        """
        for train_idx, test_idx, field_name in self.iter_lofo_cv():
            train_df = self.df.iloc[train_idx]
            test_df = self.df.iloc[test_idx]
            
            # Fit Y scaler on train data only
            y_scaler = StandardScaler()
            y_scaler.fit(train_df[[self.target_col]])

            train_dataset = PatchDataset(
                df=train_df, 
                target_col=self.target, 
                patches_dir=patches_dir, 
                patch_size=patch_size, 
                augment=True,
                y_scaler=y_scaler # Pass fitted scaler
            )
            
            test_dataset = PatchDataset(
                df=test_df, 
                target_col=self.target, 
                patches_dir=patches_dir, 
                patch_size=patch_size, 
                augment=False,
                y_scaler=y_scaler # Use same scaler
            )
            
            # Using torch imports inside scope to avoid failing if not installed
            from torch.utils.data import DataLoader
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=num_workers,
                pin_memory=True
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers,
                pin_memory=True
            )
            
            yield train_loader, test_loader, field_name

# Optional PyTorch imports for PatchDataset
try:
    import torch
    from torch.utils.data import Dataset
    import torchvision.transforms.functional as TF
    import random
    
    class PatchDataset(Dataset):
        """
        PyTorch Dataset for spatial patches. 
        Loads .npy patches and returns them along with targets.
        """
        def __init__(self, df: pd.DataFrame, target_col: str, patches_dir: str, 
                     patch_size: int = 32, augment: bool = False, y_scaler=None):
            """
            Args:
                df: DataFrame containing metadata (must have grid_id, and target_col)
                target_col: Name of the target variable to predict
                patches_dir: Directory where patch_GRIDID.npy files are saved
                patch_size: Desired patch output size (e.g. 32 or 64). Center cropped or padded.
                augment: Whether to apply random spatial augmentations (flips, rotations).
                y_scaler: Pre-fitted StandardScaler for targets. If None, fits on current df.
            """
            self.df = df.dropna(subset=[target_col]).reset_index(drop=True)
            self.target_col = target_col
            self.patches_dir = patches_dir
            self.patch_size = patch_size
            self.augment = augment
            
            # Standardize targets
            if y_scaler is not None:
                self.y_scaler = y_scaler
                self.targets = self.y_scaler.transform(self.df[[self.target_col]].values).flatten()
            else:
                self.y_scaler = StandardScaler()
                self.targets = self.y_scaler.fit_transform(self.df[[self.target_col]].values).flatten()

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            # Use the original dataframe index to find the unique patch
            orig_idx = self.df.index[idx]
            y = self.targets[idx]
            
            patch_path = os.path.join(self.patches_dir, f"patch_idx_{orig_idx}.npy")
            if os.path.exists(patch_path):
                # Shape: [Channels, H, W]
                patch = np.load(patch_path)
            else:
                # If missing, return a zeroed out tensor and handle gracefully or raise
                patch = np.zeros((13, self.patch_size, self.patch_size), dtype=np.float32)

            # Center crop to target size
            C, H, W = patch.shape
            if H != self.patch_size or W != self.patch_size:
                dh = H - self.patch_size
                dw = W - self.patch_size
                
                if dh > 0 and dw > 0:
                    top = dh // 2
                    left = dw // 2
                    patch = patch[:, top:top+self.patch_size, left:left+self.patch_size]
                else:
                    # If the patch is somehow smaller, we pad it.
                    pad_h = max(0, -dh)
                    pad_w = max(0, -dw)
                    pad_top = pad_h // 2
                    pad_bottom = pad_h - pad_top
                    pad_left = pad_w // 2
                    pad_right = pad_w - pad_left
                    patch = np.pad(patch, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
                    
            patch_tensor = torch.from_numpy(patch).float()
            
            # Simple Data Augmentation
            if self.augment:
                if random.random() > 0.5:
                    patch_tensor = TF.hflip(patch_tensor)
                if random.random() > 0.5:
                    patch_tensor = TF.vflip(patch_tensor)
                # Random 90-degree rotations
                k = random.randint(0, 3)
                if k > 0:
                    # rotate 90 * k degrees using rot90
                    patch_tensor = torch.rot90(patch_tensor, k, [1, 2])

            return patch_tensor, torch.tensor(y, dtype=torch.float32)

except ImportError:
    pass

if __name__ == "__main__":
    print("Testing data loader...")
    loader = SpatialDataLoader(target="ph", scale_features=True)
    X, y, fields = loader.get_data()
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Number of unique fields: {len(np.unique(fields))}")
    print(f"Features: {loader.get_feature_names()[:3]}...")
    
    cv_gen = loader.iter_lofo_cv()
    train_idx, test_idx, field_name = next(cv_gen)
    print(f"First fold -> test_field: {field_name}, train_size: {len(train_idx)}, test_size: {len(test_idx)}")
    print("Data loader test passed.")
