# Things I copied from our TrajFM on 3/17/25
import pandas as pd
from sklearn.model_selection import train_test_split

UKN_CATEGORY = 10

def get_train_val_test_split(data: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, random_seed: int = 42) -> tuple:
    """Splits dataset into train, validation, and test while masking category attributes for train/val."""
    # Get unique POI IDs
    pois = data["Venue ID"].unique()

    # Separate sets
    train_pois, test_pois = train_test_split(pois, test_size=test_size, random_state=random_seed)
    train_pois, val_pois = train_test_split(train_pois, test_size=val_size / (1 - test_size), random_state=random_seed)
    
    train_masked = data.copy()
    val_masked = data.copy()
    
    # Mask category by assigning -1 value
    train_masked.loc[train_masked["Venue ID"].isin(val_pois), "Category"] = UKN_CATEGORY
    train_masked.loc[train_masked["Venue ID"].isin(test_pois), "Category"] = UKN_CATEGORY
    val_masked.loc[val_masked["Venue ID"].isin(test_pois), "Category"] = UKN_CATEGORY
    
    return train_masked, val_masked, data, (train_pois, val_pois, test_pois)
