from googleapiclient.discovery import build
import json
import key
import datetime

# Your API Key
api_key = key.get_api_key()

youtube = build('youtube', 'v3', developerKey=api_key)

def get_most_viewed_videos(region_code='US', max_results=50):
    # Make an API call to get the most viewed videos
    request = youtube.videos().list(
        part="snippet,statistics",
        chart="mostPopular",
        regionCode=region_code,
        maxResults=max_results
    )
    response = request.execute()

    # Collecting titles and views
    videos = []
    for item in response.get('items', []):
        title = item['snippet']['title']
        views = item['statistics']['viewCount']
        videos.append({'title': title, 'views': views})

    return videos

# Define the timestamp
timestamp = datetime.datetime.now().strftime("%Y,%m,%d,%H:%M:%S")

# Get the most viewed videos
most_viewed_videos = get_most_viewed_videos()

# Print the result or save it as needed
for video in most_viewed_videos:
    print(f"Title: {video['title']}, Views: {video['views']}")

# Optionally, save the data to a JSON file with timestamp
json_name = '2.0/data/' + timestamp + 'most_viewed_videos.json'
with open(json_name, 'w') as f:
    json.dump(most_viewed_videos, f, indent=4)

print("Data has been saved to " + json_name)
