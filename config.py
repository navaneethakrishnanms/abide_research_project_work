import os

# Base Directories
INPUT_FOLDER = "C:/abide/UCLA_1/UCLA_1"   # Path to your raw ABIDE dataset (subject folders are here)
OUTPUT_FOLDER = "C:/abide/OUTPUT"  # Path where PNG images will be saved

# Multiprocessing Configuration
MAX_WORKERS = 4  # Change this to scale parallel processing

# Processing Flags
SAVE_ANAT = True
SAVE_REST = True

# PNG Configuration
# 0 to 9, where 0 is no compression and 9 is maximum compression.
# We use lossless PNG, but compression level can be adjusted. Default in cv2 is 3.
PNG_COMPRESSION = 3
