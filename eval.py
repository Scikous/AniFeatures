import torch
from data_preprocessor import preprocess_image
from nn import AnimeTagger

def anifeatures_eval(model, tags, model_name="anime_tagger.pth"):
    # Load the model for evaluation
    model.load_state_dict(torch.load(f'models/{model_name}'))
    model.eval()

    # Make predictions
    def predict_tags(model, image_tensor, threshold=0.5):
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = outputs.squeeze().numpy()
            predicted_tags = [tags[i] for i, prob in enumerate(probabilities) if prob > threshold]
        return predicted_tags
    image_path = "images\\84a168e2e0d0814bcc9665be6dac1cf4.png"

    # Preprocess the image
    image_tensor = preprocess_image(image_path)

    predicted_tags = predict_tags(model, image_tensor)
    print("Predicted tags:", predicted_tags)


def main():
    # Load tags
    tags_file = 'tags.csv'
    model = AnimeTagger(num_tags)

if __name__ == "__main__":
    main()
