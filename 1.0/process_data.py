
import json

def process_large_jsonl(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Convert the string line to a dictionary
            data = json.loads(line)
            # Extract the desired information, providing a default value if the key is not found
            processed_data = {
                'title': data.get('title', 'No Title'),
                'view_count': data.get('view_count', 0)  # Assuming 0 as a default value for missing view_count
            }
            # Convert the processed data back to a JSON string and write to the output file
            outfile.write(json.dumps(processed_data) + '\n')

# Make sure to replace the file paths with the correct ones for your environment.
process_large_jsonl('/hdd/YT_Dataset/_raw_yt_metadata.jsonl', '/hdd/YT_Dataset/processed_file.jsonl')
