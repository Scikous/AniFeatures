from utils import db_to_csv, image_validator, image_mover, tags_getter, csv_to_csv

def main():
    sqlite_loc = 'DanbooruDownloader-master\\DanbooruDownloader\\bin\\Debug\\net6.0-windows\\dataset\\danbooru.sqlite'
    csv_file_path = "dataset/metadata.csv"
    csv_from_file_path = "dataset/temp.csv"
    tags_to_drop_file = 'dataset/tags_to_drop.txt'
    images_source_dir = "dataset/images"# Replace with your actual source directory
    images_destination_dir = "dataset/images2" 
    tags_to_drop = tags_getter(tags_to_drop_file)

    image_validator(images_source_dir)
    image_mover(images_source_dir, images_destination_dir)
    db_to_csv(sqlite_loc, csv_file_path,images_source_dir, tags_to_drop)
    # for merging a custom csv file of images and their tags with the main file
    csv_to_csv(csv_from_file_path, csv_file_path)

if __name__ == "__main__":
    main()
