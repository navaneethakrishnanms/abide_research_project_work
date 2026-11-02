import numpy as np
from pathlib import Path
from typing import Tuple, Union

def normalize_image(image_data: np.ndarray) -> np.ndarray:
    """
    Normalizes the image data to 0-255 range and converts it to uint8.
    """
    # Remove NaN or Inf
    image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    
    if max_val - min_val == 0:
        return np.zeros_like(image_data, dtype=np.uint8)
    
    normalized = ((image_data - min_val) / (max_val - min_val)) * 255.0
    return normalized.astype(np.uint8)

def calculate_folder_size_and_count(folder_path: Union[str, Path], extension: str = ".png") -> Tuple[int, float]:
    """
    Calculates the total number of files with a given extension and their total size in MB.
    """
    path = Path(folder_path)
    if not path.exists():
        return 0, 0.0

    count = 0
    total_size_bytes = 0

    for file in path.rglob(f"*{extension}"):
        if file.is_file():
            count += 1
            total_size_bytes += file.stat().st_size

    total_size_mb = total_size_bytes / (1024 * 1024)
    return count, total_size_mb
