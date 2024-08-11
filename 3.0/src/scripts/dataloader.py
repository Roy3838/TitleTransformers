
import json
from tqdm import tqdm


# DATA LOADER 

def parse_jsonl_optimized(filepath, num_lines_to_import=None):

    titles = []
    view_counts = []
    
    with open(filepath, 'r', encoding='utf-8') as file:
        for i, line in enumerate(tqdm(file, desc="Processing")):
    
            # Check if the specified number of lines has been reached (if specified)
            if (num_lines_to_import is not None and i >= num_lines_to_import):
                break
    
            # Parse the current line
            data = json.loads(line)
   
            # Extract and store the title and view count
            titles.append(data['title'])
            view_counts.append(data['view_count'])

    return titles, view_counts

