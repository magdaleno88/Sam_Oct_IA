"""Processor for Kaggle EyePACS diabetic retinopathy dataset."""

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

import cv2
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from sam_ml.preprocessing.base import DatasetProcessor
from sam_ml.preprocessing.utils import apply_ceced_bgr_3ch, apply_clahe_bgr

if TYPE_CHECKING:
    from PIL import Image


class KaggleEyePACSProcessor(DatasetProcessor):
    """
    Processor for Kaggle EyePACS diabetic retinopathy dataset.
    
    This processor loads the dataset from Hugging Face and processes it:
    - Loads from Hugging Face dataset: bumbledeep/eyepacs
    - Applies CLAHE and CECED preprocessing
    - Saves images in TensorFlow-compatible directory structure
    """
    
    def __init__(
        self,
        raw_dir: Path = None,
        processed_dir: Path = None,
        dataset_name: str = "eyepacs_dataset",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> None:
        """
        Initialize the dataset processor.
        
        Args:
            raw_dir: Not used for Hugging Face datasets (kept for compatibility)
            processed_dir: Directory for processed dataset output
            dataset_name: Name of the processed dataset
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            random_seed: Random seed for reproducibility
        """
        # Set default paths if not provided
        if raw_dir is None:
            raw_dir = Path("data/raw") / dataset_name
        if processed_dir is None:
            processed_dir = Path("data/processed")
        
        super().__init__(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            dataset_name=dataset_name,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed
        )
    
    @property
    def supported_dataset_name(self) -> str:
        """Return the dataset name this processor supports."""
        return "eyepacs_dataset"
    
    @staticmethod
    def pad_to_square(
        img_bgr: np.ndarray,
        pad_value: tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """
        Pad the image to make it square using black (or custom color) borders.
        This prevents aspect ratio distortion when resizing.

        Args:
            img_bgr: Input image (H, W, 3)
            pad_value: Color value for padding, default black (0,0,0)

        Returns:
            A square BGR image padded to max(H, W)
        """
        h, w = img_bgr.shape[:2]
        if h == w:
            return img_bgr

        size = max(h, w)
        top = (size - h) // 2
        bottom = size - h - top
        left = (size - w) // 2
        right = size - w - left

        squared = cv2.copyMakeBorder(
            img_bgr,
            top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_value
        )
        return squared
    
    def extract_raw_data(self) -> Tuple[Path, Path]:
        """
        Load dataset from Hugging Face.
        
        Returns:
            Tuple of (train_dir, test_dir) - kept for compatibility but not used
        """
        print("=" * 60)
        print("Step 1: Loading dataset from Hugging Face...")
        print("=" * 60)
        
        print("Loading bumbledeep/eyepacs dataset...")
        dataset = load_dataset("bumbledeep/eyepacs", split="train")
        
        print(f"Loaded {len(dataset)} samples from Hugging Face")
        
        # Store dataset for later use
        self._dataset = dataset
        
        # Return dummy paths for compatibility
        return Path(""), Path("")
    
    def load_labels(self) -> Dict[str, int]:
        """
        Load labels from the Hugging Face dataset.
        
        Uses the 'label_code' field (integer 0-4) from the dataset instead of
        the 'label' field (string) to ensure numeric folder names.
        
        Returns:
            Dictionary mapping sample index to label (0-4)
        """
        print("\n" + "=" * 60)
        print("Step 2: Extracting labels from dataset...")
        print("=" * 60)
        
        labels: Dict[str, int] = {}
        
        for i, sample in enumerate(tqdm(self._dataset, desc="Extracting labels")):
            # Use label_code (integer 0-4) instead of label (string) for numeric folders
            label = sample["label_code"]
            labels[str(i)] = label
        
        print(f"Extracted {len(labels)} labels from dataset")
        return labels
    
    def create_directory_structure(self) -> Dict[str, Path]:
        """
        Create the directory structure for processed dataset.
        
        Uses numeric label folders (0, 1, 2, 3, 4) for TensorFlow compatibility.
        
        Returns:
            Dictionary with paths to created directories
        """
        print("\n" + "=" * 60)
        print("Step 3: Creating directory structure...")
        print("=" * 60)
        
        dataset_dir = self.processed_dir / self.dataset_name
        
        # Use numeric labels for TensorFlow compatibility
        label_folders = ["0", "1", "2", "3", "4"]
        
        # Create directory structure
        dirs: Dict[str, Path] = {}
        
        for channel in ["CLAHE", "CECED"]:
            for split in ["train", "val", "test"]:
                for label_folder in label_folders:
                    dir_path = dataset_dir / channel / split / label_folder
                    dir_path.mkdir(parents=True, exist_ok=True)
                    dirs[f"{channel}_{split}_{label_folder}"] = dir_path
        
        print(f"Created directory structure at {dataset_dir}")
        return dirs
    
    def split_dataset(
        self,
        image_paths: List[Path],
        labels: Dict[str, int]
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            image_paths: Not used (kept for compatibility)
            labels: Dictionary mapping sample index to label
            
        Returns:
            Tuple of (train, val, test) lists, each containing (index, label) tuples
        """
        np.random.seed(self.random_seed)
        
        # Create list of (index, label) tuples
        index_label_pairs: List[Tuple[int, int]] = [
            (int(idx), label) for idx, label in labels.items()
        ]
        
        # Shuffle
        np.random.shuffle(index_label_pairs)
        
        # Split
        total = len(index_label_pairs)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)
        
        train_data = index_label_pairs[:train_end]
        val_data = index_label_pairs[train_end:val_end]
        test_data = index_label_pairs[val_end:]
        
        print(f"Split dataset: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def _process_and_save_image(
        self,
        pil_image: "Image.Image",
        output_path: Path,
        preprocessing_fn: Callable[[np.ndarray], np.ndarray],
        label: int,
        index: int,
        target_size: Tuple[int, int] = (224, 224)
    ) -> None:
        """
        Process a single image and save it to the appropriate directory.
        
        Processing pipeline:
        1. Convert PIL Image to BGR format
        2. Apply preprocessing (CLAHE or CECED)
        3. Pad image to square to avoid aspect ratio distortion
        4. Resize to target_size (224×224 for CECED, 299×299 for CLAHE)
        5. Save as JPEG format
        
        Ensures images are:
        - JPEG format
        - Padded to square before resizing (prevents distortion)
        - Resized to target_size (224×224 for CECED, 299×299 for CLAHE)
        - 3-channel RGB/BGR
        
        Args:
            pil_image: PIL Image object in RGB format
            output_path: Path where to save the processed image (must end with .jpg)
            preprocessing_fn: Function to apply preprocessing (CLAHE or CECED)
            label: DR severity label (0-4)
            index: Sample index for naming
            target_size: Target image size (width, height). Default: (224, 224) for CECED
        """
        # Convert PIL Image (RGB) to OpenCV BGR array
        img_rgb = np.array(pil_image)
        
        # Ensure image is 3-channel RGB
        if len(img_rgb.shape) == 2:
            # Grayscale image, convert to RGB
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        elif img_rgb.shape[2] == 4:
            # RGBA image, convert to RGB
            img_rgb = img_rgb[:, :, :3]
        
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Apply preprocessing
        processed_img = preprocessing_fn(img_bgr)
        
        # Ensure processed image is 3-channel
        if len(processed_img.shape) == 2:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        elif processed_img.shape[2] == 1:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        
        # Pad to square before resizing to avoid distortions
        processed_img = self.pad_to_square(processed_img, pad_value=(0, 0, 0))
        
        # Resize to target size
        processed_img = cv2.resize(processed_img, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Ensure output path has .jpg extension
        if output_path.suffix.lower() not in ['.jpg', '.jpeg']:
            output_path = output_path.with_suffix('.jpg')
        
        # Save processed image as JPEG with high quality
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), processed_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    def process_dataset(self) -> None:
        """
        Main method to process the dataset from raw to processed format.
        
        This method orchestrates the full processing pipeline:
        1. Load dataset from Hugging Face
        2. Extract labels
        3. Create directory structure
        4. Split dataset
        5. Process images with CLAHE and CECED
        6. Pad images to square to avoid aspect ratio distortion
        7. Resize images to target sizes (299×299 for CLAHE, 224×224 for CECED)
        8. Save processed images as JPEG format
        
        All output images are guaranteed to be:
        - JPEG format (.jpg extension)
        - Padded to square before resizing (prevents distortion)
        - Correct size: 299×299 for CLAHE (InceptionV3), 224×224 for CECED (VGG16)
        - 3-channel RGB/BGR format
        """
        # Step 1: Load dataset from Hugging Face
        self.extract_raw_data()
        
        # Step 2: Load labels
        labels = self.load_labels()
        
        # Step 3: Create directory structure
        self.create_directory_structure()
        
        # Step 4: Split dataset
        print("\n" + "=" * 60)
        print("Step 4: Splitting dataset...")
        print("=" * 60)
        
        train_data, val_data, test_data = self.split_dataset([], labels)
        
        # Step 5: Process and save images
        print("\n" + "=" * 60)
        print("Step 5: Processing and saving images...")
        print("=" * 60)
        
        dataset_dir = self.processed_dir / self.dataset_name
        
        # Image sizes: CLAHE -> 299×299 (InceptionV3), CECED -> 224×224 (VGG16)
        CLAHE_SIZE = (299, 299)
        CECED_SIZE = (224, 224)
        
        # Process training images
        # Note: str(label) creates numeric folders ("0", "1", "2", "3", "4") for TensorFlow compatibility
        print(f"\nProcessing {len(train_data)} training images...")
        for idx, (sample_idx, label) in enumerate(tqdm(train_data, desc="Processing training images")):
            sample = self._dataset[sample_idx]
            pil_image = sample["image"]
            
            # Process with CLAHE (299×299) - saves to numeric folder (0-4)
            clahe_output = dataset_dir / "CLAHE" / "train" / str(label) / f"img_{sample_idx:05d}.jpg"
            self._process_and_save_image(pil_image, clahe_output, apply_clahe_bgr, label, sample_idx, target_size=CLAHE_SIZE)
            
            # Process with CECED (224×224) - saves to numeric folder (0-4)
            ceced_output = dataset_dir / "CECED" / "train" / str(label) / f"img_{sample_idx:05d}.jpg"
            self._process_and_save_image(pil_image, ceced_output, apply_ceced_bgr_3ch, label, sample_idx, target_size=CECED_SIZE)
        
        # Process validation images
        print(f"\nProcessing {len(val_data)} validation images...")
        for idx, (sample_idx, label) in enumerate(tqdm(val_data, desc="Processing validation images")):
            sample = self._dataset[sample_idx]
            pil_image = sample["image"]
            
            # Process with CLAHE (299×299)
            clahe_output = dataset_dir / "CLAHE" / "val" / str(label) / f"img_{sample_idx:05d}.jpg"
            self._process_and_save_image(pil_image, clahe_output, apply_clahe_bgr, label, sample_idx, target_size=CLAHE_SIZE)
            
            # Process with CECED (224×224)
            ceced_output = dataset_dir / "CECED" / "val" / str(label) / f"img_{sample_idx:05d}.jpg"
            self._process_and_save_image(pil_image, ceced_output, apply_ceced_bgr_3ch, label, sample_idx, target_size=CECED_SIZE)
        
        # Process test images
        print(f"\nProcessing {len(test_data)} test images...")
        for idx, (sample_idx, label) in enumerate(tqdm(test_data, desc="Processing test images")):
            sample = self._dataset[sample_idx]
            pil_image = sample["image"]
            
            # Process with CLAHE (299×299)
            clahe_output = dataset_dir / "CLAHE" / "test" / str(label) / f"img_{sample_idx:05d}.jpg"
            self._process_and_save_image(pil_image, clahe_output, apply_clahe_bgr, label, sample_idx, target_size=CLAHE_SIZE)
            
            # Process with CECED (224×224)
            ceced_output = dataset_dir / "CECED" / "test" / str(label) / f"img_{sample_idx:05d}.jpg"
            self._process_and_save_image(pil_image, ceced_output, apply_ceced_bgr_3ch, label, sample_idx, target_size=CECED_SIZE)
        
        print("\n" + "=" * 60)
        print("Dataset processing complete!")
        print("=" * 60)
        print(f"Processed dataset saved to: {dataset_dir}")
        print(f"\nDirectory structure:")
        print(f"  {dataset_dir}/CLAHE/train/{{0,1,2,3,4}}/")
        print(f"  {dataset_dir}/CLAHE/val/{{0,1,2,3,4}}/")
        print(f"  {dataset_dir}/CLAHE/test/{{0,1,2,3,4}}/")
        print(f"  {dataset_dir}/CECED/train/{{0,1,2,3,4}}/")
        print(f"  {dataset_dir}/CECED/val/{{0,1,2,3,4}}/")
        print(f"  {dataset_dir}/CECED/test/{{0,1,2,3,4}}/")

