from utils import db_to_csv, image_validator, image_mover, tags_getter
def main():
    sqlite_loc = 'DanbooruDownloader-master\\DanbooruDownloader\\bin\\Debug\\net6.0-windows\\dataset\\danbooru.sqlite'
    csv_file_path = "dataset/metadata.csv"
    tags_to_drop_file = 'dataset/tags_to_drop.txt'
    images_source_dir = "dataset/images3"#"DanbooruDownloader-master\\DanbooruDownloader\\bin\Debug\\net6.0-windows\\dataset\\images"  # Replace with your actual source directory
    images_destination_dir = "dataset/images2" 
    tags_to_drop = tags_getter(tags_to_drop_file)

    image_validator(images_source_dir)
    db_to_csv(sqlite_loc, images_source_dir, csv_file_path, tags_to_drop)
    image_mover(images_source_dir, images_destination_dir)

if __name__ == "__main__":
    main()