#!/usr/bin/env python3
"""
Script to download and prepare Rust code from the ammarnasr/the-stack-rust-clean dataset
for nanochat pretraining.

This is an open-access dataset on HuggingFace Hub.

Usage:
    python scripts/prepare_rust_data.py

Output:
    data/rust_code_train.parquet - Parquet file with 'text' column for nanochat pretraining
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_huggingface_auth():
    """Check if user is authenticated with HuggingFace (optional for open-access datasets)."""
    from huggingface_hub import whoami
    
    try:
        user = whoami()
        print(f"✓ Authenticated as: {user.get('name', 'unknown')}")
        return True
    except Exception as e:
        print(f"Note: Not authenticated to HuggingFace Hub (optional for open-access datasets)")
        print(f"  Reason: {e}")
        return True  # Continue anyway since dataset is open-access


def load_and_analyze_dataset(num_samples=5):
    """Load a small sample to analyze the dataset structure."""
    from datasets import load_dataset
    
    print("\n" + "=" * 60)
    print("Loading sample dataset to analyze structure...")
    print("=" * 60)
    
    # Load a small sample to see the structure
    ds = load_dataset(
        "ammarnasr/the-stack-rust-clean",
        split=f"train[:{num_samples}]",
        trust_remote_code=True
    )
    
    print(f"\nDataset info:")
    print(f"  Number of samples: {len(ds)}")
    print(f"  Column names: {ds.column_names}")
    
    print("\nFirst example (all fields):")
    example = ds[0]
    for key, value in example.items():
        if isinstance(value, str):
            preview = value[:150] + "..." if len(value) > 150 else value
        else:
            preview = str(value)
        print(f"  {key}: {preview}")
    
    # Get dataset info for statistics
    print("\n" + "=" * 60)
    print("Getting dataset statistics...")
    print("=" * 60)
    
    # Get the full dataset info (without loading all data)
    ds_info = load_dataset(
        "ammarnasr/the-stack-rust-clean",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    print("\nDataset statistics (ammarnasr/the-stack-rust-clean):")
    print("  Language: Rust")
    print("  Note: This is an open-access dataset")
    
    return ds


def prepare_training_data(
    output_path: str = "data/rust_code_train.parquet",
    max_samples: int = 100000,
    min_file_size: int = 100,
    max_file_size: int = 500000,
) -> None:
    """
    Download Rust code from ammarnasr/the-stack-rust-clean and convert to nanochat format.
    
    Args:
        output_path: Path to save the parquet file
        max_samples: Maximum number of samples to download
        min_file_size: Minimum file size in bytes (filter out very small files)
        max_file_size: Maximum file size in bytes (filter out very large files)
    """
    from datasets import load_dataset
    import pandas as pd
    
    print("\n" + "=" * 60)
    print("Preparing Rust code training data...")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  Output path: {output_path}")
    print(f"  Max samples: {max_samples:,}")
    print(f"  Min file size: {min_file_size} bytes")
    print(f"  Max file size: {max_file_size} bytes")
    
    # Load dataset - streaming mode to handle large dataset
    print("\nLoading dataset (streaming mode)...")
    
    ds = load_dataset(
        "ammarnasr/the-stack-rust-clean",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    # Determine the content column name by examining first sample
    print("\nDetecting content column name...")
    sample = next(iter(ds))
    possible_columns = ['content', 'code', 'text', 'source_code', 'source']
    content_column = None
    for col in possible_columns:
        if col in sample:
            content_column = col
            print(f"  Found content column: '{col}'")
            break
    if content_column is None:
        # Use first string column if none of the expected ones
        for col, val in sample.items():
            if isinstance(val, str) and len(val) > 100:
                content_column = col
                print(f"  Using column: '{col}' (detected as code content)")
                break
    if content_column is None:
        content_column = list(sample.keys())[0]
        print(f"  Using first column: '{content_column}'")
    
    # Process samples
    print("Processing samples...")
    texts = []
    total_processed = 0
    total_filtered = 0
    
    for sample in ds:
        if len(texts) >= max_samples:
            break
            
        # Get content field - detect the correct column name
        content = sample.get(content_column, "")
        
        # Filter by file size
        content_len = len(content)
        if content_len < min_file_size or content_len > max_file_size:
            total_filtered += 1
            continue
        
        # Add to our list - the 'text' column is what nanochat expects
        texts.append({"text": content})
        total_processed += 1
        
        if total_processed % 1000 == 0:
            print(f"  Processed: {total_processed:,} / {max_samples:,}")
    
    print(f"\nProcessing complete!")
    print(f"  Total processed: {total_processed:,}")
    print(f"  Filtered (size): {total_filtered:,}")
    print(f"  Final count: {len(texts):,}")
    
    if not texts:
        print("\nNo samples collected! Check filters or dataset availability.")
        return
    
    # Create DataFrame and save as parquet
    print("\nCreating parquet file...")
    df = pd.DataFrame(texts)
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    df.to_parquet(output_path, index=False)
    
    print(f"\n✓ Saved to: {output_path}")
    print(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Number of samples: {len(df):,}")
    print(f"  Column: 'text' (contains Rust source code)")
    
    # Show sample
    print("\nSample (first 200 chars of first entry):")
    print(f"  {df['text'].iloc[0][:200]}...")


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Prepare Rust code training data")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--max-samples", type=int, default=100000, help="Maximum number of samples")
    parser.add_argument("--min-size", type=int, default=100, help="Minimum file size in bytes")
    parser.add_argument("--max-size", type=int, default=500000, help="Maximum file size in bytes")
    parser.add_argument("--output", type=str, default="data/rust_code_train.parquet", help="Output path")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Nanochat - Prepare Rust Code Training Data")
    print("=" * 60)
    print("\nDataset: ammarnasr/the-stack-rust-clean (HuggingFace)")
    print("Language: Rust")
    print("Output: Parquet file with 'text' column for pretraining")
    
    # Check authentication (optional for open-access datasets)
    check_huggingface_auth()
    
    # First, analyze dataset structure with small sample
    sample_ds = load_and_analyze_dataset(num_samples=3)
    
    # Ask user if they want to proceed (unless -y flag is provided)
    if not args.yes:
        print("\n" + "=" * 60)
        response = input("Proceed with downloading training data? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            sys.exit(0)
    else:
        print("\nAuto-proceeding with -y flag...")
    
    # Prepare training data
    prepare_training_data(
        output_path=args.output,
        max_samples=args.max_samples,
        min_file_size=args.min_size,
        max_file_size=args.max_size,
    )
    
    print("\n" + "=" * 60)
    print("Done! The parquet file is ready for nanochat pretraining.")
    print("=" * 60)


if __name__ == "__main__":
    main()
