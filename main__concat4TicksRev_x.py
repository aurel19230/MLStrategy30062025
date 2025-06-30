import os
import csv
import time
import pandas as pd
import standardFunc_sauv
from standardFunc_sauv import timestamp_to_date_utc,date_to_timestamp_utc
# Entrées utilisateur
folder_path = 'C:/Users/aulac/OneDrive/Documents/Trading/VisualStudioProject/Sierra chart/xTickReversal/data_simuNew/4_0_4TP_1SL/merge'
output_file_name = '_4TicksRev.csv'

def check_files(folder_path):
    # List files in the folder
    files = [f for f in os.listdir(folder_path) if
             f.endswith('.csv') and '_' in f and f.split('_')[-1].split('.')[0].isdigit()]

    # Verify that files have sequential suffixes
    file_indices = sorted([int(f.split('_')[-1].split('.')[0]) for f in files])
    if file_indices != list(range(min(file_indices), max(file_indices) + 1)):
        print("Files are not sequentially numbered. Ensure they follow the format _0, _1, _2, etc.")
        return False

    # Sort files based on indices
    files_sorted = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return files_sorted


def process_files():
    files = check_files(folder_path)
    if not files:
        return

    start_date = None
    end_date = None
    total_rows = 0

    output_file = os.path.join(folder_path, output_file_name)

    # Check if prefixed_output file exists, if so, clear its content and write 0
    if os.path.exists(output_file):
        with open(output_file, 'w') as outfile:
            outfile.write('0\n')
        print(f"File {output_file} existed and was cleared and written with 0.")

    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=';')

        for i, file in enumerate(files):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r') as infile:
                reader = csv.reader(infile, delimiter=';')
                rows = list(reader)

                if i == 0:
                    start_date = timestamp_to_date_utc(int(rows[1][0]))
                    print(
                        f"{file} (index {i}): {timestamp_to_date_utc(int(rows[1][0]))} - {timestamp_to_date_utc(int(rows[-1][0]))} | Rows: {len(rows)}")
                else:
                    print(
                        f"{file} (index {i}): {timestamp_to_date_utc(int(rows[0][0]))} - {timestamp_to_date_utc(int(rows[-1][0]))} | Rows: {len(rows)}")

                end_date = timestamp_to_date_utc(int(rows[-1][0]))
                total_rows += len(rows)

                if i == 0:
                    writer.writerows(rows)
                else:
                    writer.writerows(rows[0:])

    start_date = start_date.replace('-', '').replace(':', '').replace(' ', '_')
    end_date = end_date.replace('-', '').replace(':', '').replace(' ', '_')

    prefixed_output = os.path.join(folder_path, f"{start_date[:8]}_{end_date[:8]}{output_file_name}")

    # If the prefixed_output file already exists, remove it before renaming
    if os.path.exists(prefixed_output):
        os.remove(prefixed_output)

    os.rename(output_file, prefixed_output)
    print(f"Fichiers fusionnés dans {prefixed_output}")

    # Verify the number of lines in the merged file
    with open(prefixed_output, 'r') as merged_file:
        merged_lines_count = sum(1 for line in merged_file)
    print(f"Total rows in merged file {prefixed_output}: {merged_lines_count}")

    # Ensure the number of rows matches the total rows counted from individual files
    if total_rows == merged_lines_count:
        print(f"\033[92mNumber of lines matches: {total_rows} rows.\033[0m")
    else:
        print(
            f"\033[91mNumber of lines does not match: {total_rows} rows counted, but {merged_lines_count} rows in the merged file.\033[0m")
        exit()

    # Load the merged file into a DataFrame and display it
    merged_df = pd.read_csv(prefixed_output, sep=';')
    print(merged_df)


if __name__ == "__main__":
    process_files()
