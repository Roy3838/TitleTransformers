from googleapiclient.discovery import build
import json
import key
import datetime

# Your API Key
api_key = key.get_api_key()


# Initialize the YouTube API client
youtube = build('youtube', 'v3', developerKey=api_key)

def get_most_viewed_videos(max_results, regions):
    all_videos = []

    for region_code in regions:
        print(f"Fetching videos for region: {region_code}")
        next_page_token = None
        region_videos = 0  # Track the number of videos fetched per region

        while region_videos < max_results:
            print(f"Fetching next {min(max_results - region_videos, 50)} videos, remaining for region {region_code}: {max_results - region_videos}")
            request = youtube.videos().list(
                part="snippet,statistics",
                chart="mostPopular",
                regionCode=region_code,
                maxResults=min(max_results - region_videos, 50),
                pageToken=next_page_token
            )
            response = request.execute()

            # Check and print the number of items fetched
            items_fetched = len(response.get('items', []))
            print(f"Items fetched for region {region_code}: {items_fetched}")

            # Collecting titles and views
            for item in response['items']:
                title = item['snippet']['title']
                views = item['statistics']['viewCount']
                all_videos.append({'title': title, 'views': views, 'region': region_code})

            # Update the next_page_token and reduce the max_results
            next_page_token = response.get('nextPageToken')
            region_videos += items_fetched
            print(f"Next page token for region {region_code}: {next_page_token}")

            if not next_page_token:
                print(f"No more pages to fetch for region {region_code}.")
                break

    return all_videos

# Define the timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# List of region codes
regions = ['US', 'CA', 'GB', 'DE', 'IN']  # Add or remove regions as needed

# Scrape size per region
scrape_size_per_region = 500  # Change this to the number of videos you want to scrape per region

# Get the most viewed videos
most_viewed_videos = get_most_viewed_videos(max_results=scrape_size_per_region, regions=regions)

# Print the result or save it as needed
for video in most_viewed_videos:
    print(f"Region: {video['region']}, Title: {video['title']}, Views: {video['views']}")

# Optionally, save the data to a JSON file with timestamp
json_name = '2.0/data/' + timestamp + '-N-' + str(scrape_size_per_region * len(regions)) + '-most_viewed_videos.json'
with open(json_name, 'w') as f:
    json.dump(most_viewed_videos, f, indent=4)

print("Data has been saved to " + json_name)