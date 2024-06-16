import torch
from model import AniFeatures
from utils import tags_getter, preprocess_image, image_loader

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
       # print(f"image_filename: {img} Predicted tags: {predicted_tags}\n")
    return image_tags

def main():
    # Load tags
    images = image_loader('dataset/images/')
    tags_file = 'dataset/metadata_tags.txt'
    tags = tags_getter(tags_file)
    model = AniFeatures(num_tags=len(tags))
    results = anifeatures_eval(model, tags, images, threshold=0.7, model_name="anime_tagger.pth")
    print(results)
    
if __name__ == "__main__":
    main()
