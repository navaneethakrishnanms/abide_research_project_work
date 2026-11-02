import os
import cv2
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from logger import logger
from utils import normalize_image

class SubjectConverter:
    """
    Class to handle the conversion of a single subject's MRI/fMRI data to PNG images.
    """
    def __init__(self, subject_path: Path, output_base: Path):
        self.subject_path = subject_path
        self.output_base = output_base
        self.subject_id = subject_path.name
        
        # Stats tracked per subject
        self.converted_anat = 0
        self.converted_rest = 0
        self.pngs_created = 0
        self.errors = 0

    def convert_3d_to_png(self, img_data: np.ndarray, output_dir: Path) -> int:
        """
        Converts a 3D volume (X,Y,Z) to PNG slices along the Z axis.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        
        # Typically image is (X, Y, Z). We want slices along Z.
        for z in range(img_data.shape[2]):
            slice_data = img_data[:, :, z]
            
            # Normalize
            slice_img = normalize_image(slice_data)
            
            # Transpose or rotate if necessary (skipping here to preserve raw data orientation)
            # Typically nibabel loads data where (X,Y) might need a 90 deg rotation for typical viewing,
            # but we preserve original orientation as requested (just raw array values).
            
            out_file = output_dir / f"slice_{z:03d}.png"
            
            # Save using OpenCV, grayscale, lossless
            cv2.imwrite(
                str(out_file), 
                slice_img, 
                [cv2.IMWRITE_PNG_COMPRESSION, config.PNG_COMPRESSION]
            )
            count += 1
            
        return count

    def convert_4d_to_png(self, img_data: np.ndarray, output_dir: Path) -> int:
        """
        Converts a 4D volume (X,Y,Z,T) to PNG slices along the Z axis for each T.
        """
        count = 0
        
        for t in range(img_data.shape[3]):
            time_dir = output_dir / f"time_{t:03d}"
            time_dir.mkdir(parents=True, exist_ok=True)
            
            time_volume = img_data[:, :, :, t]
            
            for z in range(time_volume.shape[2]):
                slice_data = time_volume[:, :, z]
                
                # Normalize
                slice_img = normalize_image(slice_data)
                
                out_file = time_dir / f"slice_{z:03d}.png"
                
                cv2.imwrite(
                    str(out_file), 
                    slice_img, 
                    [cv2.IMWRITE_PNG_COMPRESSION, config.PNG_COMPRESSION]
                )
                count += 1
                
        return count

    def process(self) -> Dict[str, int]:
        """
        Scans the subject folder, finds relevant NIfTI files, and converts them.
        """
        # Look for anat and rest
        # Typical structure: subject_id / session_1 / anat_1 / mprage.nii.gz
        
        nifti_files = list(self.subject_path.rglob("*.nii.gz")) + list(self.subject_path.rglob("*.nii"))
        
        for file_path in nifti_files:
            try:
                file_name = file_path.name.lower()
                is_anat = "mprage" in file_name or "anat" in str(file_path).lower()
                is_rest = "rest" in file_name or "func" in str(file_path).lower()
                
                if is_anat and not config.SAVE_ANAT:
                    continue
                if is_rest and not config.SAVE_REST:
                    continue
                    
                if not is_anat and not is_rest:
                    # Unknown file type based on simple heuristic, skip or process?
                    # Instructions imply mprage.nii.gz for anat and rest.nii.gz for rest.
                    continue
                
                # Load image
                img = nib.load(str(file_path))
                img_data = img.get_fdata()
                
                if is_anat and len(img_data.shape) == 3:
                    out_dir = self.output_base / "anat" / self.subject_id
                    png_count = self.convert_3d_to_png(img_data, out_dir)
                    self.pngs_created += png_count
                    self.converted_anat += 1
                    
                elif is_rest and len(img_data.shape) >= 4:
                    out_dir = self.output_base / "rest" / self.subject_id
                    png_count = self.convert_4d_to_png(img_data, out_dir)
                    self.pngs_created += png_count
                    self.converted_rest += 1
                else:
                    logger.warning(f"Unexpected shape {img_data.shape} for {file_path}. Skipping.")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                self.errors += 1
                
        return {
            "anat_converted": self.converted_anat,
            "rest_converted": self.converted_rest,
            "pngs_created": self.pngs_created,
            "errors": self.errors
        }

class AbideConverter:
    """
    Main orchestrator for converting the entire ABIDE dataset.
    """
    def __init__(self):
        self.input_dir = Path(config.INPUT_FOLDER)
        self.output_dir = Path(config.OUTPUT_FOLDER)
        
        self.total_subjects = 0
        self.processed_subjects = 0
        self.total_anat = 0
        self.total_rest = 0
        self.total_pngs = 0
        self.total_errors = 0

    def run(self):
        if not self.input_dir.exists():
            logger.error(f"Input directory does not exist: {self.input_dir}")
            return

        # Find all subject folders (assuming they are direct children of input_dir)
        # Or you can adapt this to find folders based on the structure.
        # ABIDE folders are typically just directories inside the input folder
        subject_folders = [f for f in self.input_dir.iterdir() if f.is_dir()]
        self.total_subjects = len(subject_folders)
        
        logger.info(f"Found {self.total_subjects} subjects in {self.input_dir}")
        
        if self.total_subjects == 0:
            return

        # Process in parallel
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            # Submit tasks
            future_to_subject = {
                executor.submit(SubjectConverter(subj_path, self.output_dir).process): subj_path.name
                for subj_path in subject_folders
            }
            
            # Progress bar
            with tqdm(total=self.total_subjects, desc="Converting Subjects", unit="subj") as pbar:
                for future in as_completed(future_to_subject):
                    subject_id = future_to_subject[future]
                    try:
                        results = future.result()
                        self.total_anat += results["anat_converted"]
                        self.total_rest += results["rest_converted"]
                        self.total_pngs += results["pngs_created"]
                        self.total_errors += results["errors"]
                    except Exception as exc:
                        logger.error(f"Subject {subject_id} generated an exception: {exc}")
                        self.total_errors += 1
                    finally:
                        self.processed_subjects += 1
                        pbar.set_postfix({
                            "Subj": subject_id,
                            "PNGs": self.total_pngs,
                            "Errs": self.total_errors
                        })
                        pbar.update(1)
