from pathlib import Path
import requests

def download_file(url):
    path = Path('data_original')
    path.mkdir(parents=True, exist_ok=True)
    
    filename = Path(url.split('/')[-1].split('?')[0])
    filepath = path / filename
    
    response = requests.get(url)
    
    if response.status_code == 200:
        filepath.write_bytes(response.content)
        print(f"File downloaded successfully: {filepath}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

urls = ["https://www.ers.usda.gov/webdocs/DataFiles/100812/Top%205%20Agricultural%20Exports%20by%20State.csv?v=1388", 
        "https://www.ers.usda.gov/webdocs/DataFiles/100812/Top%205%20Agricultural%20Imports%20by%20State.csv?v=1388"]

for url in urls:
    download_file(url)
