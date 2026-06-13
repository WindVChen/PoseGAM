#!/usr/bin/env python3
"""
Script to combine multiple CSV files from BOP benchmark results.
Handles merging CSV files with the same structure.
"""

import os
import csv
import argparse
from pathlib import Path


def combine_csv_files(input_files, output_file, skip_duplicates=True):
    """
    Combine multiple CSV files into one.
    
    Args:
        input_files: List of input CSV file paths
        output_file: Output CSV file path
        skip_duplicates: If True, skip duplicate rows based on scene_id, im_id, obj_id
    """
    if not input_files:
        print("No input files provided!")
        return
    
    # Read header from first file
    with open(input_files[0], 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        print(f"CSV columns: {fieldnames}")
    
    # Track seen rows to avoid duplicates
    seen_rows = set()
    total_rows = 0
    duplicate_rows = 0
    
    # Write combined CSV
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for input_file in input_files:
            if not os.path.exists(input_file):
                print(f"Warning: File not found: {input_file}")
                continue
                
            print(f"Processing: {input_file}")
            
            with open(input_file, 'r', newline='') as infile:
                reader = csv.DictReader(infile)
                
                # Verify columns match
                if reader.fieldnames != fieldnames:
                    print(f"Warning: Column mismatch in {input_file}")
                    print(f"  Expected: {fieldnames}")
                    print(f"  Got: {reader.fieldnames}")
                    continue
                
                file_rows = 0
                for row in reader:
                    total_rows += 1
                    
                    # Create unique key for duplicate detection
                    if skip_duplicates:
                        row_key = (row.get('scene_id', ''), 
                                  row.get('im_id', ''), 
                                  row.get('obj_id', ''))
                        
                        if row_key in seen_rows:
                            duplicate_rows += 1
                            continue
                        
                        seen_rows.add(row_key)
                    
                    writer.writerow(row)
                    file_rows += 1
                
                print(f"  Added {file_rows} rows")
    
    print(f"\nSummary:")
    print(f"  Total rows processed: {total_rows}")
    print(f"  Duplicate rows skipped: {duplicate_rows}")
    print(f"  Rows written: {total_rows - duplicate_rows}")
    print(f"  Output file: {output_file}")


def find_csv_files(directory, pattern):
    """
    Find CSV files matching a pattern in a directory.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern (e.g., "*_results_*.csv")
    
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    return [str(f) for f in files]


def main():
    parser = argparse.ArgumentParser(
        description="Combine the per-rank CSV files written by a parallel test_BOP_benchmark.py run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
test_BOP_benchmark.py with --total_ranks N writes one CSV per rank:
    <output_name>_<dataset>_results_rank<r>_of_<N>.csv
This tool merges them (de-duplicating by scene_id/im_id/obj_id) into a single file.

Name the output following the BOP toolkit convention <method>_<dataset>-test_*.csv so it
can be fed directly to the BOP evaluation toolkit, e.g.:

  # merge all ranks for one dataset
  python combine_csv.py -d /path/to/results \\
      -p "posegam_ycbv_results_rank*_of_*.csv" \\
      -o posegam_ycbv-test_combined.csv

  # or pass explicit files
  python combine_csv.py -i rank0.csv rank1.csv -o posegam_ycbv-test_combined.csv
        """
    )

    parser.add_argument('-i', '--input', nargs='+', help='Input CSV files')
    parser.add_argument('-d', '--directory', help='Directory to search for CSV files', default='.')
    parser.add_argument('-p', '--pattern', default='*_results_rank*_of_*.csv',
                       help='Glob pattern for the per-rank CSV files written by test_BOP_benchmark.py')
    parser.add_argument('-o', '--output', default='posegam_ycbv-test_combined.csv',
                       help='Output CSV file. Follow the BOP toolkit naming <method>_<dataset>-test_*.csv')
    parser.add_argument('--allow-duplicates', action='store_false',
                       help='Allow duplicate rows (default: skip duplicates based on scene_id, im_id, obj_id)')
    
    args = parser.parse_args()
    
    # Determine input files
    if args.input:
        input_files = args.input
    elif args.directory:
        input_files = find_csv_files(args.directory, args.pattern)
        if not input_files:
            print(f"No CSV files found matching pattern '{args.pattern}' in {args.directory}")
            return
        print(f"Found {len(input_files)} CSV files:")
        for f in input_files:
            print(f"  - {f}")
        print()
    else:
        print("Error: Must provide either --input files or --directory")
        parser.print_help()
        return
    
    # Combine files
    combine_csv_files(input_files, args.output, skip_duplicates=not args.allow_duplicates)


if __name__ == "__main__":
    main()
