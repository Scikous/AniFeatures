import torch
from model import AniFeatures
from utils import tags_getter, preprocess_image, image_loader, tagged_images_to_json

#use a model to evaluate on image(s) and get the corresponding tags for each one
def anifeatures_eval(model, tags, images, threshold=0.5, model_name="anime_tagger.pth"):
    # Load the model for evaluation
    model.load_state_dict(torch.load(f'models/{model_name}'))
    model.eval()
    image_tags = []
    # Make predictions
    def predict_tags(model, image_tensor, threshold=threshold):
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = outputs.squeeze().numpy()
            predicted_tags = [tags[i] for i, prob in enumerate(probabilities) if prob > threshold]
        return predicted_tags
    
    for img in images:
        # Preprocess the image
        image_tensor = preprocess_image(img)
        #print(image_tensor)
        predicted_tags = predict_tags(model, image_tensor)
        image_tags.append((img, predicted_tags))
    return image_tags


def main():
    # Load tags
    images = image_loader('dataset/images4/')
    tags_file = 'dataset/metadata_tags.txt'
    tagged_imgs_csv = 'dataset/tagged_imgs.json'
    tags = tags_getter(tags_file)
    model = AniFeatures(num_tags=len(tags))
    results = anifeatures_eval(model, tags, images, threshold=0.45,model_name="anime_tagger2.pth")
    tagged_images_to_json(results, tagged_imgs_csv)
if __name__ == "__main__":
    main()
