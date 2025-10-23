import os
import sys
import pandas as pd

def chunk_and_save_csv(input_file, chunk_size, output_folder, output_prefix):
    """Splits a CSV file into chunks and saves them in a new directory."""

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    chunk_iter = pd.read_csv(input_file, chunksize=chunk_size, encoding='utf-16', sep='\t')

    for i, chunk in enumerate(chunk_iter):
        output_file = os.path.join(output_folder, f"{output_prefix}_{i+1}.csv")
        chunk.to_csv(output_file, index=False, header=True)

        if i == 0:
            print(f"Processing: {input_file}")
            print("Columns:", chunk.columns)

        print(f"Chunk {i+1} shape: {chunk.shape}")
        print(f"Saved {output_file}")

    # Remove original CSV after processing
    os.remove(input_file)
    print(f"Removed original file: {input_file}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python chunk_csv.py <csv_file> <chunk_size> <output_folder> <output_prefix>")
        sys.exit(1)

    input_file = sys.argv[1]
    chunk_size = int(sys.argv[2])
    output_folder = sys.argv[3]
    output_prefix = sys.argv[4]

    chunk_and_save_csv(input_file, chunk_size, output_folder, output_prefix)
