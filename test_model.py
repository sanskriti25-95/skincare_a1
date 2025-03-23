import tensorflow as tf
import numpy as np
from PIL import Image
import os

def predict_skin_type(image_path, model):
    """Predict skin type from an image"""
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Make prediction
    prediction = model.predict(img_array)
    skin_types = ['dry', 'normal', 'oily']
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class] * 100
    
    return skin_types[predicted_class], confidence, prediction[0]

def test_model():
    # Load the model
    print("Loading model...")
    model = tf.keras.models.load_model('skin_classifier_model.h5')
    
    # Test with validation images
    dataset_path = "dataset/validation"
    total_correct = 0
    total_images = 0
    
    print("\nTesting model with validation images:")
    print("-" * 50)
    
    for skin_type in ['dry', 'normal', 'oily']:
        skin_type_dir = os.path.join(dataset_path, skin_type)
        if os.path.exists(skin_type_dir):
            images = os.listdir(skin_type_dir)[:5]  # Test first 5 images of each type
            correct = 0
            
            print(f"\nTesting {skin_type} skin images:")
            for image in images:
                image_path = os.path.join(skin_type_dir, image)
                predicted_type, confidence, probabilities = predict_skin_type(image_path, model)
                
                # Print results
                print(f"\nImage: {image}")
                print(f"True type: {skin_type}")
                print(f"Predicted type: {predicted_type}")
                print(f"Confidence: {confidence:.2f}%")
                print(f"Probabilities: Dry: {probabilities[0]*100:.2f}%, "
                      f"Normal: {probabilities[1]*100:.2f}%, "
                      f"Oily: {probabilities[2]*100:.2f}%")
                
                if predicted_type == skin_type:
                    correct += 1
                    total_correct += 1
                total_images += 1
            
            print(f"\nAccuracy for {skin_type} skin: {(correct/len(images))*100:.2f}%")
    
    print("\nOverall Results:")
    print(f"Total Accuracy: {(total_correct/total_images)*100:.2f}%")

if __name__ == "__main__":
    test_model() 