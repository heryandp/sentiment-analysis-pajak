import os
import json
import csv
import glob

def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            # For lists, we'll just join them with a semicolon or keep as string
            # But here we assume simple lists or ignore deep nesting in lists for now
            out[name[:-1]] = str(x)
        else:
            if isinstance(x, str):
                # Replace newlines and carriage returns with a space
                x = x.replace('\n', ' ').replace('\r', ' ')
            out[name[:-1]] = x

    flatten(y)
    return out

def main():
    source_dir = 'raw_content'
    output_file = 'combined_data.csv'
    
    # Get all json files
    json_files = glob.glob(os.path.join(source_dir, '*.json'))
    
    if not json_files:
        print("No JSON files found in", source_dir)
        return

    all_data = []
    all_keys = set()

    print(f"Found {len(json_files)} JSON files. Processing...")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Flatten the data
                flat_data = flatten_json(data)
                all_data.append(flat_data)
                all_keys.update(flat_data.keys())
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Define the specific columns we want to keep
    target_columns = ['title', 'link', 'snippet', 'date_clean', 'full_content']

    print(f"Writing {len(all_data)} records to {output_file} with columns: {target_columns}...")
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=target_columns, extrasaction='ignore')
            writer.writeheader()
            for row in all_data:
                writer.writerow(row)
        print("Done!")
    except Exception as e:
        print(f"Error writing CSV: {e}")

if __name__ == "__main__":
    main()
