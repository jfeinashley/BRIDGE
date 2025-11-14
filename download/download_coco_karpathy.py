#!/usr/bin/env python3
"""
Download script for COCO Karpathy split used by BLIP.
This script downloads the exact COCO setup used in the BLIP project.
"""

import os
import json
import argparse
from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import tarfile
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path, desc=None):
    """Download a file from a URL with progress bar."""
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return
    
    print(f"Downloading: {desc if desc else output_path}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urlretrieve(url, output_path, reporthook=t.update_to)
    print(f"Downloaded: {output_path}")


def download_karpathy_annotations(output_dir):
    """Download Karpathy split annotation files."""
    annotations_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(annotations_dir, exist_ok=True)
    
    # URLs for Karpathy splits from BLIP
    karpathy_files = {
        'coco_karpathy_train.json': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json',
        'coco_karpathy_val.json': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
        'coco_karpathy_test.json': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json',
        'coco_karpathy_val_gt.json': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
        'coco_karpathy_test_gt.json': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'
    }
    
    print("\n" + "="*50)
    print("Downloading Karpathy COCO Annotations")
    print("="*50)
    
    for filename, url in karpathy_files.items():
        output_path = os.path.join(annotations_dir, filename)
        download_url(url, output_path, desc=filename)
    
    print("\n✓ All Karpathy annotations downloaded successfully!")
    return annotations_dir


def download_coco_images(output_dir, year='2017'):
    """Download COCO images."""
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print(f"Downloading COCO {year} Images")
    print("="*50)
    
    # COCO image URLs
    if year == '2017':
        image_urls = {
            'train2017.zip': 'http://images.cocodataset.org/zips/train2017.zip',
            'val2017.zip': 'http://images.cocodataset.org/zips/val2017.zip',
            'test2017.zip': 'http://images.cocodataset.org/zips/test2017.zip'
        }
    elif year == '2014':
        image_urls = {
            'train2014.zip': 'http://images.cocodataset.org/zips/train2014.zip',
            'val2014.zip': 'http://images.cocodataset.org/zips/val2014.zip', 
            'test2014.zip': 'http://images.cocodataset.org/zips/test2014.zip'
        }
    else:
        raise ValueError(f"Year {year} not supported. Use 2014 or 2017.")
    
    for filename, url in image_urls.items():
        zip_path = os.path.join(images_dir, filename)
        
        # Download if not exists
        if not os.path.exists(zip_path):
            print(f"\nDownloading {filename} (~13-18GB per file)...")
            print("This will take a while depending on your internet connection.")
            download_url(url, zip_path, desc=filename)
        
        # Extract if not already extracted
        extract_dir = zip_path.replace('.zip', '')
        if not os.path.exists(extract_dir):
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(images_dir)
            print(f"✓ Extracted to {extract_dir}")
        else:
            print(f"✓ {extract_dir} already exists, skipping extraction")
    
    print("\n✓ All COCO images downloaded and extracted successfully!")
    return images_dir


def verify_setup(output_dir):
    """Verify that the downloaded files match BLIP's expected structure."""
    print("\n" + "="*50)
    print("Verifying Setup")
    print("="*50)
    
    annotations_dir = os.path.join(output_dir, 'annotations')
    images_dir = os.path.join(output_dir, 'images')
    
    # Check annotations
    required_annotations = [
        'coco_karpathy_train.json',
        'coco_karpathy_val.json', 
        'coco_karpathy_test.json',
        'coco_karpathy_val_gt.json',
        'coco_karpathy_test_gt.json'
    ]
    
    print("\nChecking annotations:")
    for ann_file in required_annotations:
        path = os.path.join(annotations_dir, ann_file)
        if os.path.exists(path):
            # Load and check JSON is valid
            with open(path, 'r') as f:
                data = json.load(f)
                print(f"  ✓ {ann_file}: {len(data)} entries")
        else:
            print(f"  ✗ Missing: {ann_file}")
    
    # Check for image directories
    print("\nChecking image directories:")
    for subdir in os.listdir(images_dir) if os.path.exists(images_dir) else []:
        if os.path.isdir(os.path.join(images_dir, subdir)):
            num_images = len([f for f in os.listdir(os.path.join(images_dir, subdir)) 
                            if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  ✓ {subdir}: {num_images} images")
    
    print("\n" + "="*50)
    print("Setup Complete!")
    print("="*50)
    print(f"\nYour COCO dataset is ready at: {output_dir}")
    print("\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"    ├── annotations/  # Karpathy split JSON files")
    print(f"    └── images/       # COCO image files")
    print("\nYou can now use this dataset with BLIP-style dataloaders!")


def main():
    parser = argparse.ArgumentParser(description='Download COCO dataset with Karpathy splits (BLIP setup)')
    parser.add_argument('--output_dir', type=str, default='./coco_karpathy',
                        help='Output directory for the dataset (default: ./coco_karpathy)')
    parser.add_argument('--annotations_only', action='store_true',
                        help='Download only annotations without images')
    parser.add_argument('--year', type=str, default='2014', choices=['2014', '2017'],
                        help='COCO dataset year (default: 2014, as BLIP uses COCO 2014)')
    parser.add_argument('--skip_verification', action='store_true',
                        help='Skip verification step')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print("COCO Karpathy Dataset Downloader (BLIP Setup)")
    print("="*50)
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    
    # Download annotations
    annotations_dir = download_karpathy_annotations(args.output_dir)
    
    # Download images unless --annotations_only
    if not args.annotations_only:
        images_dir = download_coco_images(args.output_dir, year=args.year)
    else:
        print("\n⚠ Skipping image download (--annotations_only flag set)")
        print("You'll need to download COCO images separately for the dataset to work.")
    
    # Verify setup
    if not args.skip_verification:
        verify_setup(args.output_dir)
    
    print("\n✨ Done!")
    
    # Print usage instructions
    print("\n" + "="*50)
    print("Usage Instructions")
    print("="*50)
    print("\nTo use this dataset in your code:")
    print("```python")
    print("# For BLIP-style dataloader")
    print(f"image_root = '{os.path.abspath(args.output_dir)}/images/'")
    print(f"ann_root = '{os.path.abspath(args.output_dir)}/annotations/'")
    print("```")
    
    if args.annotations_only:
        print("\n⚠ Remember to download COCO images before using the dataset!")
        print(f"Run again without --annotations_only to download images.")


if __name__ == '__main__':
    main()
