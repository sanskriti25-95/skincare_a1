from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import json
import base64
import io

app = Flask(__name__)

class SkinAnalysis:
    def __init__(self):
        self.model = tf.keras.models.load_model('skin_classifier_model.h5')
        with open('product_recommendations.json', 'r') as f:
            self.recommendations = json.load(f)
        
        self.questions = [
            ("basic_info", "age", "What is your age?"),
            ("basic_info", "gender", "What is your gender? (Female/Male/Other)"),
            ("basic_info", "climate", "What climate do you live in?\n1. Tropical/Humid\n2. Hot and Dry\n3. Moderate\n4. Cold and Dry\n5. Coastal/Humid"),
            ("skin_characteristics", "skin_feel", "How does your skin feel after washing?\n1. Tight and uncomfortable\n2. Slightly dry but comfortable\n3. Normal and comfortable\n4. Slightly oily in T-zone\n5. Oily all over"),
            ("skin_characteristics", "morning_skin", "How does your skin look in the morning?\n1. Dry and flaky\n2. Normal\n3. Slightly oily\n4. Very oily\n5. Combination"),
            ("skin_concerns", "primary_concern", "What is your primary skin concern?\n1. Dryness and flaking\n2. Oiliness and shine\n3. Acne and breakouts\n4. Aging/Fine lines\n5. None"),
            ("lifestyle_factors", "water_intake", "How many glasses of water do you drink daily?\n1. Less than 4\n2. 4-6\n3. 6-8\n4. More than 8"),
            ("product_preferences", "budget_range", "What's your monthly skincare budget (₹)?\n1. 500-1000\n2. 1000-2000\n3. 2000-3000\n4. 3000+")
        ]
        
    def validate_face_image(self, image):
        """
        Validate that the uploaded image contains a clearly visible face
        Returns: (is_valid, message)
        """
        try:
            # Convert PIL Image to CV2 format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Load multiple face detection classifiers
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
            face_cascade_alt2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Enhance image contrast
            gray = cv2.equalizeHist(gray)
            
            # Try different detection parameters
            faces = []
            scale_factors = [1.1, 1.15, 1.2]
            min_neighbors_options = [3, 4, 5]
            
            for scale in scale_factors:
                for min_neighbors in min_neighbors_options:
                    # Try with different classifiers
                    faces_default = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=scale,
                        minNeighbors=min_neighbors,
                        minSize=(60, 60),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    faces_alt = face_cascade_alt.detectMultiScale(
                        gray,
                        scaleFactor=scale,
                        minNeighbors=min_neighbors,
                        minSize=(60, 60),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    faces_alt2 = face_cascade_alt2.detectMultiScale(
                        gray,
                        scaleFactor=scale,
                        minNeighbors=min_neighbors,
                        minSize=(60, 60),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    if len(faces_default) > 0:
                        faces = faces_default
                        break
                    elif len(faces_alt) > 0:
                        faces = faces_alt
                        break
                    elif len(faces_alt2) > 0:
                        faces = faces_alt2
                        break
                
                if len(faces) > 0:
                    break
            
            if len(faces) == 0:
                return False, "No face detected in the image. Please upload a clear photo of your face."
            elif len(faces) > 1:
                return False, "Multiple faces detected. Please upload a photo with just your face."
            
            return True, "Valid face image detected."
            
        except Exception as e:
            print(f"Error in face detection: {str(e)}")  # For debugging
            return False, f"Error processing image. Please try a different photo. Error: {str(e)}"

    def analyze_skin_type(self, image, answers):
        """Analyze skin type based on image and questionnaire"""
        # Get model prediction
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        prediction = self.model.predict(img_array)
        skin_types = ['dry', 'normal', 'oily']
        
        # Initialize scoring system
        skin_scores = {
            'dry': prediction[0][0] * 30,    # Model contributes 30%
            'normal': prediction[0][1] * 30,
            'oily': prediction[0][2] * 30
        }
        
        # Map answers to scores
        feel_map = {
            "1": {'dry': 15, 'normal': 0, 'oily': 0},
            "2": {'dry': 10, 'normal': 5, 'oily': 0},
            "3": {'dry': 0, 'normal': 15, 'oily': 0},
            "4": {'dry': 0, 'normal': 5, 'oily': 10},
            "5": {'dry': 0, 'normal': 0, 'oily': 15}
        }
        
        # Apply questionnaire scores
        if 'skin_feel' in answers['skin_characteristics']:
            feel_scores = feel_map[answers['skin_characteristics']['skin_feel']]
            for skin_type in skin_scores:
                skin_scores[skin_type] += feel_scores[skin_type]
        
        # Normalize scores
        total_score = sum(skin_scores.values())
        if total_score > 0:
            for skin_type in skin_scores:
                skin_scores[skin_type] = (skin_scores[skin_type] / total_score) * 100
        
        # Get final skin type
        final_type = max(skin_scores, key=skin_scores.get)
        confidence = skin_scores[final_type]
        
        return final_type, confidence, skin_scores

analyzer = SkinAnalysis()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/validate_image', methods=['POST'])
def validate_image():
    image_data = request.json['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    is_valid, message = analyzer.validate_face_image(image)
    return jsonify({'valid': is_valid, 'message': message})

@app.route('/get_question', methods=['POST'])
def get_question():
    question_index = int(request.json['current_question'])
    if question_index < len(analyzer.questions):
        category, key, question = analyzer.questions[question_index]
        return jsonify({
            'question': question,
            'category': category,
            'key': key,
            'is_last': question_index == len(analyzer.questions) - 1
        })
    return jsonify({'done': True})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    answers = data['answers']
    skin_type, confidence, scores = analyzer.analyze_skin_type(image, answers)
    
    recommendations = analyzer.recommendations[skin_type]
    
    return jsonify({
        'skin_type': skin_type,
        'confidence': confidence,
        'recommendations': recommendations
    })

if __name__ == '__main__':
    app.run(debug=True) 