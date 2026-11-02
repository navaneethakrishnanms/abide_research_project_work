import time
from logger import logger
import config
from converter import AbideConverter
from utils import calculate_folder_size_and_count

def main():
    logger.info("="*50)
    logger.info("Starting ABIDE Dataset Conversion")
    logger.info(f"Input Folder:  {config.INPUT_FOLDER}")
    logger.info(f"Output Folder: {config.OUTPUT_FOLDER}")
    logger.info(f"Max Workers:   {config.MAX_WORKERS}")
    logger.info("="*50)

    start_time = time.time()
    
    converter = AbideConverter()
    converter.run()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    logger.info("Calculating final output folder size...")
    png_count, size_mb = calculate_folder_size_and_count(config.OUTPUT_FOLDER, extension=".png")

    logger.info("="*50)
    logger.info("CONVERSION SUMMARY")
    logger.info(f"Subjects Processed:      {converter.processed_subjects}/{converter.total_subjects}")
    logger.info(f"MRI Files Converted:     {converter.total_anat}")
    logger.info(f"fMRI Files Converted:    {converter.total_rest}")
    logger.info(f"PNG Images Created:      {converter.total_pngs}")
    logger.info(f"Errors Encountered:      {converter.total_errors}")
    logger.info(f"Execution Time:          {execution_time:.2f} seconds")
    logger.info(f"Output Folder Size:      {size_mb:.2f} MB")
    logger.info(f"Total PNG Count Verify:  {png_count}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
