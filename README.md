# ABIDE Dataset Converter

A robust, multi-threaded Python project to convert the official ABIDE MRI/fMRI dataset (`.nii.gz` files) into grayscale, lossless PNG images. 

## Features
- Preserves the original folder hierarchy by organizing outputs neatly into `anat/` and `rest/` directories.
- Fully automated conversion of both 3D Structural MRI (`mprage`) and 4D Functional MRI (`rest`).
- Fast multi-threading using `ThreadPoolExecutor`.
- Comprehensive error handling and logging to `conversion.log`.
- Progress tracking with `tqdm`.
- Memory efficient design.

## Prerequisites
- Windows 11
- Python 3.11+

## Installation

1. Ensure you have Python installed.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Before running the tool, modify the paths in `config.py` to match your local setup:

- `INPUT_FOLDER`: Path to the raw ABIDE dataset.
- `OUTPUT_FOLDER`: Path where the PNG images will be saved.
- `MAX_WORKERS`: Number of parallel threads (default is 4).

## How to Run

Simply execute the `main.py` script:

```bash
python main.py
```

## Expected Folder Structure

**Input (Example):**
```
INPUT/
├── 0051263/
│   └── session_1/
│       ├── anat_1/
│       │   └── mprage.nii.gz
│       └── rest_1/
│           └── rest.nii.gz
├── 0051264/
...
```

**Output (Example):**
```
OUTPUT/
├── anat/
│   ├── 0051263/
│   │   ├── slice_000.png
│   │   └── slice_001.png
│   └── 0051264/
│       ...
└── rest/
    ├── 0051263/
    │   ├── time_000/
    │   │   ├── slice_000.png
    │   │   └── slice_001.png
    │   └── time_001/
    │       ...
    └── 0051264/
        ...
```
# abide_research_project_work
