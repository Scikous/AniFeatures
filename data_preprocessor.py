from utils import db_to_csv, image_validator, image_mover
def main():
    sqlite_loc = 'DanbooruDownloader-master\\DanbooruDownloader\\bin\\Debug\\net6.0-windows\\dataset\\danbooru.sqlite'
    csv_filename = "metadata2.csv"
    images_source_dir = "images"#"DanbooruDownloader-master\\DanbooruDownloader\\bin\Debug\\net6.0-windows\\dataset\\images"  # Replace with your actual source directory
    images_destination_dir = "images2" 

    image_validator(images_source_dir)
    db_to_csv(sqlite_loc, csv_filename)
    image_mover(images_source_dir, images_destination_dir)

if __name__ == "__main__":
    main()