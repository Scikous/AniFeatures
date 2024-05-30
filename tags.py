import requests
import time

#retrieves all available danbooru tags and writesthe mto a .txt file
def danbooru_tags_retriever(tags_get_url, output_filename):
    # bad_categories = [1,3,4,5]
    # good_categories = [0]
    # Open a file to write
    with open(output_filename, mode='w', encoding='utf-8') as file:
        # Loop through pages 1 to 1000
        for page in range(1, 1001):
            # Update the URL with the current page
            url = f'{tags_get_url}&page={page}'
            # Fetch the JSON data
            response = requests.get(url)
            
            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()
                
                # Break the loop if the data is empty (no more tags to fetch)
                if not data:
                    print(f'No more data found at page {page}. Stopping.', flush=True)
                    break
                
                # Write the data
                for item in data:
                    file.write(f"{item['name']}\n")
                    
                
                # Explicitly flush the data to the file
                file.flush()
            else:
                print(f'Failed to fetch data for page {page}. HTTP Status Code: {response.status_code}', flush=True)
                break

            print(f'Page {page} processed.', flush=True)
            # Sleep for 1 second so we don't DDOS Danbooru too much
            time.sleep(1)
        print(f'Data has been written to {output_filename}', flush=True)


def main():
    tags_get_url = 'https://danbooru.donmai.us/tags.json?limit=1000&search[hide_empty]=yes&search[is_deprecated]=no&search[order]=count' #&search[category]=0
    output_filename = 'tags_test.txt'
    danbooru_tags_retriever(tags_get_url, output_filename)


if __name__ == "__main__":
    # Base URL without the page parameter
    main()

