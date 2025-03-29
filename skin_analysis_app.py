import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
import cv2
import io

class SkinAnalysisApp:
    def __init__(self):
        try:
            self.model = tf.keras.models.load_model('skin_classifier_model.h5')
        except:
            st.error("Model file not found!")
            st.stop()
            
        try:
            with open('product_recommendations.json', 'r') as f:
                self.recommendations = json.load(f)
        except:
            st.error("Recommendations file not found!")
            st.stop()
    
    def get_skin_assessment_questions(self):
        questions = {
            "basic_info": {
                "age": st.slider("What is your age?", 12, 80, 25),
                "gender": st.selectbox("Gender", ["Female", "Male", "Other"]),
                "location_climate": st.selectbox(
                    "What climate do you live in?", 
                    ["Tropical/Humid", "Hot and Dry", "Moderate", "Cold and Dry", "Coastal/Humid"]
                ),
                "occupation": st.selectbox(
                    "Nature of your work",
                    ["Indoor/Office", "Outdoor Work", "Mixed Indoor-Outdoor", "Student", "Other"]
                )
            },
            "skin_characteristics": {
                "skin_feel": st.selectbox(
                    "How does your skin feel after washing?",
                    [
                        "Tight and uncomfortable",
                        "Slightly dry but comfortable",
                        "Normal and comfortable",
                        "Slightly oily in T-zone",
                        "Oily all over"
                    ]
                ),
                "morning_skin": st.selectbox(
                    "How does your skin look in the morning?",
                    [
                        "Dry and flaky",
                        "Normal",
                        "Slightly oily",
                        "Very oily",
                        "Combination (oily T-zone, dry cheeks)"
                    ]
                ),
                "pore_size": st.selectbox(
                    "How would you describe your pores?",
                    [
                        "Almost invisible",
                        "Small and fine",
                        "Visible on nose only",
                        "Visible on T-zone",
                        "Large and visible across face"
                    ]
                ),
                "skin_texture": st.selectbox(
                    "How would you describe your skin texture?",
                    [
                        "Rough and flaky",
                        "Slightly rough",
                        "Smooth",
                        "Bumpy with small bumps",
                        "Uneven with large pores"
                    ]
                )
            },
            "skin_concerns": {
                "primary_concern": st.selectbox(
                    "What is your primary skin concern?",
                    [
                        "Dryness and flaking",
                        "Oiliness and shine",
                        "Acne and breakouts",
                        "Pigmentation/Dark spots",
                        "Aging/Fine lines",
                        "Sensitivity/Redness",
                        "Uneven texture",
                        "Large pores",
                        "None"
                    ]
                ),
                "sensitivity": st.selectbox(
                    "How sensitive is your skin?",
                    [
                        "Not sensitive at all",
                        "Slightly sensitive to new products",
                        "Moderately sensitive (occasional reactions)",
                        "Very sensitive (frequent reactions)",
                        "Extremely sensitive (reacts to most products)"
                    ]
                ),
                "acne_frequency": st.selectbox(
                    "How often do you experience breakouts?",
                    [
                        "Never",
                        "Rarely (few times a year)",
                        "Occasionally (monthly)",
                        "Frequently (weekly)",
                        "Constantly"
                    ]
                ),
                "specific_issues": st.multiselect(
                    "Select any specific skin issues you have:",
                    [
                        "Blackheads",
                        "Whiteheads",
                        "Cystic acne",
                        "Rosacea",
                        "Eczema",
                        "Melasma",
                        "Sun damage",
                        "Post-acne marks",
                        "None"
                    ]
                )
            },
            "lifestyle_factors": {
                "sun_exposure": st.selectbox(
                    "Daily sun exposure",
                    [
                        "Minimal (indoor most of the day)",
                        "Low (1-2 hours outdoors)",
                        "Moderate (2-4 hours outdoors)",
                        "High (4+ hours outdoors)"
                    ]
                ),
                "sunscreen_use": st.selectbox(
                    "How often do you use sunscreen?",
                    [
                        "Never",
                        "Only when outdoors for long",
                        "Most days",
                        "Daily, once",
                        "Daily, with reapplication"
                    ]
                ),
                "water_intake": st.selectbox(
                    "Daily water intake",
                    [
                        "Less than 4 glasses",
                        "4-6 glasses",
                        "6-8 glasses",
                        "8-10 glasses",
                        "More than 10 glasses"
                    ]
                ),
                "sleep_quality": st.selectbox(
                    "How would you rate your sleep quality?",
                    [
                        "Poor (less than 5 hours)",
                        "Fair (5-6 hours)",
                        "Good (6-7 hours)",
                        "Very good (7-8 hours)",
                        "Excellent (8+ hours)"
                    ]
                ),
                "stress_level": st.selectbox(
                    "How would you rate your stress level?",
                    [
                        "Low",
                        "Moderate",
                        "High",
                        "Very high"
                    ]
                ),
                "diet": st.multiselect(
                    "Select items that are regular part of your diet:",
                    [
                        "Dairy products",
                        "Sugary foods",
                        "Spicy food",
                        "Processed foods",
                        "Green vegetables",
                        "Fruits",
                        "Fish/Omega-3 rich foods",
                        "Nuts and seeds"
                    ]
                )
            },
            "product_preferences": {
                "current_routine": st.text_area(
                    "Describe your current skincare routine (if any)",
                    height=100
                ),
                "product_type": st.multiselect(
                    "What type of products do you prefer?",
                    [
                        "Natural/Ayurvedic",
                        "Korean skincare",
                        "Medical/Clinical",
                        "Organic",
                        "Fragrance-free",
                        "Budget-friendly",
                        "Premium brands"
                    ]
                ),
                "budget_range": st.selectbox(
                    "Monthly skincare budget (₹)",
                    [
                        "500-1000",
                        "1000-2000",
                        "2000-3000",
                        "3000-5000",
                        "5000+"
                    ]
                ),
                "allergies": st.text_input(
                    "List any known allergies or ingredients you avoid"
                )
            }
        }
        return questions

    def analyze_skin_type(self, image, answers):
        # Get model prediction (reduce weight to 30%)
        img = Image.open(image)
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        prediction = self.model.predict(img_array)
        skin_types = ['dry', 'normal', 'oily']
        
        # Initialize scoring system
        skin_scores = {
            'dry': prediction[0][0] * 30,    # Model contributes 30% of initial score
            'normal': prediction[0][1] * 30,
            'oily': prediction[0][2] * 30
        }
        
        # Analyze skin characteristics (40% weight)
        skin_feel_map = {
            "Tight and uncomfortable": {'dry': 15, 'normal': 0, 'oily': 0},
            "Slightly dry but comfortable": {'dry': 10, 'normal': 5, 'oily': 0},
            "Normal and comfortable": {'dry': 0, 'normal': 15, 'oily': 0},
            "Slightly oily in T-zone": {'dry': 0, 'normal': 5, 'oily': 10},
            "Oily all over": {'dry': 0, 'normal': 0, 'oily': 15}
        }
        
        morning_skin_map = {
            "Dry and flaky": {'dry': 15, 'normal': 0, 'oily': 0},
            "Normal": {'dry': 0, 'normal': 15, 'oily': 0},
            "Slightly oily": {'dry': 0, 'normal': 5, 'oily': 10},
            "Very oily": {'dry': 0, 'normal': 0, 'oily': 15},
            "Combination (oily T-zone, dry cheeks)": {'dry': 5, 'normal': 10, 'oily': 5}
        }
        
        pore_size_map = {
            "Almost invisible": {'dry': 10, 'normal': 5, 'oily': 0},
            "Small and fine": {'dry': 5, 'normal': 10, 'oily': 0},
            "Visible on nose only": {'dry': 0, 'normal': 10, 'oily': 5},
            "Visible on T-zone": {'dry': 0, 'normal': 5, 'oily': 10},
            "Large and visible across face": {'dry': 0, 'normal': 0, 'oily': 10}
        }
        
        # Apply skin characteristic scores with higher weight
        feel_scores = skin_feel_map[answers['skin_characteristics']['skin_feel']]
        morning_scores = morning_skin_map[answers['skin_characteristics']['morning_skin']]
        pore_scores = pore_size_map[answers['skin_characteristics']['pore_size']]
        
        for skin_type in skin_scores:
            skin_scores[skin_type] += (feel_scores[skin_type] + 
                                     morning_scores[skin_type] + 
                                     pore_scores[skin_type])
        
        # Add strong validation checks (30% weight)
        # Check for oily indicators
        if answers['skin_concerns']['primary_concern'] == "Oiliness and shine":
            skin_scores['oily'] += 20
            skin_scores['dry'] -= 10
        
        if "Blackheads" in answers['skin_concerns']['specific_issues'] or "Whiteheads" in answers['skin_concerns']['specific_issues']:
            skin_scores['oily'] += 10
            skin_scores['dry'] -= 5
        
        # Check for dry indicators
        if answers['skin_concerns']['primary_concern'] == "Dryness and flaking":
            skin_scores['dry'] += 20
            skin_scores['oily'] -= 10
        
        if "Eczema" in answers['skin_concerns']['specific_issues']:
            skin_scores['dry'] += 10
            skin_scores['oily'] -= 5
        
        # Climate adjustments
        climate = answers['basic_info']['location_climate']
        if climate in ["Tropical/Humid", "Coastal/Humid"]:
            skin_scores['oily'] += 5
            skin_scores['dry'] -= 5
        elif climate in ["Hot and Dry", "Cold and Dry"]:
            skin_scores['dry'] += 5
            skin_scores['oily'] -= 5
        
        # Ensure no negative scores
        for skin_type in skin_scores:
            skin_scores[skin_type] = max(0, skin_scores[skin_type])
        
        # Normalize scores
        total_score = sum(skin_scores.values())
        if total_score > 0:  # Avoid division by zero
            for skin_type in skin_scores:
                skin_scores[skin_type] = (skin_scores[skin_type] / total_score) * 100
        
        # Determine final skin type with bias correction
        final_type = max(skin_scores, key=skin_scores.get)
        confidence = skin_scores[final_type]
        
        # Add combination skin detection
        if abs(skin_scores['oily'] - skin_scores['dry']) < 15:  # Close scores indicate combination
            if skin_scores['normal'] > 30:
                final_type = 'normal'  # Lean towards normal if scores are close
        
        return final_type, confidence, skin_scores

    def get_recommendations(self, skin_type, answers):
        budget_ranges = {
            "500-1000": 1000,
            "1000-2000": 2000,
            "2000-3000": 3000,
            "3000-5000": 5000,
            "5000+": 10000
        }
        max_budget = budget_ranges[answers['product_preferences']['budget_range']]
        
        # Define product database from the comprehensive list
        product_database = {
            'oily': {
                'cleansers': [
                    {
                        'name': 'Neutrogena Oil-Free Acne Wash',
                        'price': 475,
                        'key_ingredients': ['Salicylic Acid', 'Aloe'],
                        'description': 'Oil-free formula for acne-prone skin'
                    },
                    {
                        'name': 'Himalaya Purifying Neem Face Wash',
                        'price': 85,
                        'key_ingredients': ['Neem', 'Turmeric'],
                        'description': 'Natural neem-based cleanser'
                    },
                    {
                        'name': 'The Face Shop Rice Water Bright Cleansing Foam',
                        'price': 450,
                        'key_ingredients': ['Rice Extract', 'Ceramides'],
                        'description': 'Brightening and oil-control'
                    }
                ],
                'moisturizers': [
                    {
                        'name': 'Sebamed Clear Face Gel',
                        'price': 479,
                        'key_ingredients': ['Hyaluronic Acid', 'Aloe Vera'],
                        'description': 'Non-comedogenic gel moisturizer'
                    },
                    {
                        'name': 'Neutrogena Hydro Boost Water Gel',
                        'price': 850,
                        'key_ingredients': ['Hyaluronic Acid', 'Glycerin'],
                        'description': 'Oil-free hydration'
                    }
                ],
                'treatments': [
                    {
                        'name': 'The Body Shop Tea Tree Oil',
                        'price': 895,
                        'key_ingredients': ['Tea Tree Oil'],
                        'description': 'Spot treatment for acne'
                    },
                    {
                        'name': 'Minimalist Salicylic Acid 2%',
                        'price': 549,
                        'key_ingredients': ['Salicylic Acid', 'Tea Tree'],
                        'description': 'Exfoliating serum for acne'
                    }
                ]
            },
            'dry': {
                'cleansers': [
                    {
                        'name': 'Cetaphil Gentle Skin Cleanser',
                        'price': 300,
                        'key_ingredients': ['Glycerin', 'Panthenol'],
                        'description': 'Gentle non-stripping cleanser'
                    },
                    {
                        'name': 'Forest Essentials Delicate Facial Cleanser',
                        'price': 1175,
                        'key_ingredients': ['Honey', 'Aloe'],
                        'description': 'Nourishing cream cleanser'
                    }
                ],
                'moisturizers': [
                    {
                        'name': 'Bioderma Atoderm Intensive Baume',
                        'price': 1450,
                        'key_ingredients': ['Ceramides', 'Glycerin'],
                        'description': 'Rich moisturizer for very dry skin'
                    },
                    {
                        'name': 'Embryolisse Lait-Crème Concentré',
                        'price': 2200,
                        'key_ingredients': ['Shea Butter', 'Aloe'],
                        'description': 'Multi-purpose moisturizing cream'
                    }
                ]
            },
            'normal': {
                'cleansers': [
                    {
                        'name': 'Simple Kind To Skin Refreshing Facial Wash',
                        'price': 375,
                        'key_ingredients': ['Pro-Vitamin B5', 'Vitamin E'],
                        'description': 'Gentle daily cleanser'
                    },
                    {
                        'name': 'Cetaphil Gentle Skin Cleanser',
                        'price': 300,
                        'key_ingredients': ['Glycerin', 'Panthenol'],
                        'description': 'Mild non-irritating cleanser'
                    }
                ],
                'moisturizers': [
                    {
                        'name': 'Neutrogena Oil-Free Moisture Combination Skin',
                        'price': 450,
                        'key_ingredients': ['Glycerin', 'Vitamin E'],
                        'description': 'Lightweight daily moisturizer'
                    },
                    {
                        'name': 'Clinique Dramatically Different Moisturizing Gel',
                        'price': 2400,
                        'key_ingredients': ['Hyaluronic Acid', 'Glycerin'],
                        'description': 'Oil-free hydrating gel'
                    }
                ]
            }
        }

        # Filter products based on budget
        affordable_products = {}
        for category, products in product_database[skin_type].items():
            affordable_products[category] = [p for p in products if p['price'] <= max_budget]
        
        return {
            'routine': self.recommendations[skin_type]['routine'],
            'products': affordable_products,
            'home_remedies': self.recommendations[skin_type]['home_remedies'],
            'lifestyle_tips': self.get_lifestyle_tips(answers)
        }

    def display_product_recommendations(self, products, budget_range):
        st.subheader("Recommended Products Within Your Budget")
        
        for category, items in products.items():
            st.write(f"\n**{category.title()}**")
            if items:
                for product in items:
                    st.write(f"**{product['name']}**")
                    st.write(f"₹{product['price']}")
                    st.write(f"Key ingredients: {', '.join(product['key_ingredients'])}")
                    st.write(f"_{product['description']}_")
                    st.write("---")
            else:
                st.write("No products found in this category within your budget.")

    def validate_face_image(self, image):
        """
        Validate that the uploaded image contains a clearly visible face
        Returns: (is_valid, message)
        """
        try:
            # Convert PIL Image to CV2 format
            img = Image.open(image)
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
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
                    
                    # Combine detected faces
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
                # Try with reduced constraints if no face detected
                for cascade in [face_cascade, face_cascade_alt, face_cascade_alt2]:
                    faces = cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.3,
                        minNeighbors=2,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    if len(faces) > 0:
                        break
            
            if len(faces) == 0:
                return False, "No face detected in the image. Please upload a clear photo of your face."
            elif len(faces) > 1:
                return False, "Multiple faces detected. Please upload a photo with just your face."
            
            # Check if face is large enough in the image
            face_area = faces[0][2] * faces[0][3]  # width * height
            image_area = img_cv.shape[0] * img_cv.shape[1]
            face_ratio = face_area / image_area
            
            if face_ratio < 0.05:  # Reduced minimum face size requirement
                return False, "Face is too small in the image. Please upload a closer photo of your face."
            elif face_ratio < 0.1:  # Reduced warning threshold
                return True, "Warning: Face appears small in the image. Results might be less accurate."
            
            return True, "Valid face image detected."
            
        except Exception as e:
            return False, f"Error processing image. Please try a different photo. Error: {str(e)}"

    def get_lifestyle_tips(self, answers):
        """Generate lifestyle tips based on questionnaire answers"""
        lifestyle_tips = []
        
        # Water intake recommendations
        if answers['lifestyle_factors']['water_intake'] in ['Less than 4 glasses', '4-6 glasses']:
            lifestyle_tips.append("Increase water intake to at least 8 glasses per day")
        
        # Sleep recommendations
        sleep_quality = answers['lifestyle_factors']['sleep_quality']
        if sleep_quality in ['Poor (less than 5 hours)', 'Fair (5-6 hours)', 'Good (6-7 hours)']:
            lifestyle_tips.append("Try to get 7-8 hours of sleep for better skin health")
        
        # Sun exposure recommendations
        sun_exposure = answers['lifestyle_factors']['sun_exposure']
        if sun_exposure in ['Moderate (2-4 hours outdoors)', 'High (4+ hours outdoors)']:
            lifestyle_tips.append("Use sunscreen regularly and reapply every 2-3 hours when outdoors")
        
        # Diet recommendations
        diet = answers['lifestyle_factors']['diet']
        if 'Dairy products' in diet and 'Sugary foods' in diet:
            lifestyle_tips.append("Consider reducing dairy and sugar intake for better skin health")
        if not any(food in diet for food in ['Green vegetables', 'Fruits']):
            lifestyle_tips.append("Include more fruits and vegetables in your diet for skin-healthy nutrients")
        
        # Stress management
        if answers['lifestyle_factors']['stress_level'] in ['High', 'Very high']:
            lifestyle_tips.append("Consider stress-management techniques like meditation or yoga")
        
        # Climate-based recommendations
        climate = answers['basic_info']['location_climate']
        if climate in ["Tropical/Humid", "Coastal/Humid"]:
            lifestyle_tips.append("Use lightweight, non-comedogenic products in humid weather")
        elif climate in ["Hot and Dry", "Cold and Dry"]:
            lifestyle_tips.append("Focus on hydration and moisturizing in dry weather")
        
        # Occupation-based tips
        occupation = answers['basic_info']['occupation']
        if occupation in ["Outdoor Work", "Mixed Indoor-Outdoor"]:
            lifestyle_tips.append("Ensure consistent sun protection throughout the day")
        
        # Skin sensitivity tips
        if answers['skin_concerns']['sensitivity'] in ["Very sensitive", "Extremely sensitive"]:
            lifestyle_tips.append("Patch test new products and introduce them gradually")
        
        return lifestyle_tips

def main():
    st.set_page_config(page_title="Indian Skin Analysis System", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #ff4b4b;
            color: white;
        }
        .stProgress > div > div > div > div {
            background-color: #ff4b4b;
        }
        h1 {
            color: #ff4b4b;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title('🌟 Indian Skin Analysis & Recommendation System')
    
    app = SkinAnalysisApp()
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Navigate", ["Home", "About", "How It Works"])
    
    if page == "Home":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("""
            ### Welcome to Your Personalized Skin Analysis Journey! 
            
            Get customized skincare recommendations based on:
            - Advanced AI skin analysis
            - Detailed skin assessment
            - Indian skin type considerations
            - Budget-friendly product suggestions
            """)
            
            # Add image source selection
            image_source = st.radio(
                "Choose how to provide your image:",
                ["Upload Image", "Take Photo"],
                horizontal=True
            )
            
            if image_source == "Upload Image":
                uploaded_file = st.file_uploader(
                    "Upload a clear, well-lit image of your face",
                    type=['jpg', 'jpeg', 'png']
                )
                image_file = uploaded_file
            else:
                st.write("📸 Take a photo of your face")
                camera_image = st.camera_input("Take a picture")
                image_file = camera_image
                
                if camera_image:
                    st.info("""
                    Tips for a good photo:
                    - Find good lighting
                    - Keep your face centered
                    - Hold the camera at arm's length
                    - Keep a neutral expression
                    """)
            
            if image_file is not None:
                # Validate image contains a face
                try:
                    is_valid, message = app.validate_face_image(image_file)
                    
                    if not is_valid:
                        st.error(message)
                        st.stop()
                    elif "Warning" in message:
                        st.warning(message)
                    
                    # Display the image with face detection
                    img = Image.open(image_file)
                    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
                    
                    # Draw rectangle around face
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Convert back to PIL Image for display
                    img_display = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                    
                    # Display image in a container with fixed height
                    with st.container():
                        st.image(img_display, caption='Detected Face', use_column_width=True)
                    
                    # Add guidelines for better photos
                    if image_source == "Take Photo":
                        st.info("""
                        If the photo isn't clear enough:
                        - Find better lighting
                        - Make sure your face is clearly visible
                        - Try to minimize motion blur
                        - Keep the camera steady
                        """)
                    else:
                        st.info("""
                        📸 For best results:
                        - Ensure your face is well-lit
                        - Look directly at the camera
                        - Remove glasses if possible
                        - Avoid heavy makeup
                        - Keep a neutral expression
                        """)
                    
                except Exception as e:
                    st.error("Error processing image. Please try again with a different photo.")
                    st.stop()
            
            st.write("### Let's understand your skin better!")
            st.write("Please answer these questions for a more accurate analysis:")
            
            answers = app.get_skin_assessment_questions()
            
            if st.button("Analyze My Skin"):
                with st.spinner("Analyzing your skin and creating your personalized recommendations..."):
                    skin_type, confidence, confidence_levels = app.analyze_skin_type(image_file, answers)
                    
                    st.success("Analysis Complete! 🌟")
                    
                    st.subheader("Your Personalized Skin Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # More friendly way to present skin type
                        st.write(f"**Your Primary Skin Type:** {skin_type.capitalize()} ✨")
                        
                        # Add personalized description based on skin type
                        skin_descriptions = {
                            'dry': """
                            Your skin tends to feel tight and needs extra moisture. This is common and can be 
                            effectively managed with the right skincare routine. Many people with dry skin have 
                            a naturally glowing complexion when well-hydrated!
                            """,
                            'oily': """
                            Your skin produces more natural oils, which can actually be beneficial as it helps 
                            maintain youthful skin! With the right products, you can maintain a healthy balance 
                            and achieve a natural, dewy look.
                            """,
                            'normal': """
                            You have a well-balanced skin type! While your skin is generally stable, it's still 
                            important to maintain a consistent skincare routine to keep it healthy and glowing.
                            """
                        }
                        st.write(skin_descriptions[skin_type])
                        
                        # Add personalized observations based on questionnaire
                        st.write("\n**Key Observations:**")
                        observations = []
                        
                        # Climate-based observation
                        climate = answers['basic_info']['location_climate']
                        if climate in ["Tropical/Humid", "Coastal/Humid"]:
                            observations.append("Living in a humid climate means your skin needs protection from moisture loss and sun damage")
                        elif climate in ["Hot and Dry", "Cold and Dry"]:
                            observations.append("Your climate can cause moisture loss, so hydration should be a priority")
                        
                        # Lifestyle-based observations
                        if answers['lifestyle_factors']['water_intake'] in ['Less than 4 glasses', '4-6 glasses']:
                            observations.append("Increasing your water intake could help improve your skin's hydration")
                        
                        if answers['lifestyle_factors']['stress_level'] in ['High', 'Very high']:
                            observations.append("High stress levels might be affecting your skin - consider stress-management techniques")
                        
                        # Skin concerns
                        if answers['skin_concerns']['sensitivity'] in ["Very sensitive", "Extremely sensitive"]:
                            observations.append("Your skin's sensitivity means you should introduce new products gradually")
                        
                        for obs in observations:
                            st.write(f"- {obs}")
                    
                    recommendations = app.get_recommendations(skin_type, answers)
                    
                    # Recommendations section with improved presentation
                    st.subheader("Your Customized Skincare Journey 🌸")
                    
                    # Morning and Evening Routine
                    st.write("**Daily Skincare Routine**")
                    routine = recommendations['routine']
                    
                    # Handle different routine formats
                    if '\n\n' in routine:
                        routines = routine.split('\n\n')
                    else:
                        # If routine is not in expected format, split by Morning/Evening
                        morning_routine = [line for line in routine.split('\n') if 'Morning' in line or (line.startswith('1.') and not any(prev.startswith('Evening') for prev in routine.split('\n')[:routine.split('\n').index(line)]))]
                        evening_routine = [line for line in routine.split('\n') if 'Evening' in line or (line.startswith('1.') and any(prev.startswith('Evening') for prev in routine.split('\n')[:routine.split('\n').index(line)]))]
                        routines = ['\n'.join(morning_routine), '\n'.join(evening_routine)]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("🌅 **Morning Routine**")
                        if routines and len(routines) > 0:
                            morning = routines[0].replace("Morning:", "").strip()
                            for step in morning.split('\n'):
                                if step.strip():  # Only display non-empty steps
                                    st.write(f"- {step.strip()}")
                        else:
                            st.write("- Gentle cleanser\n- Moisturizer\n- Sunscreen")
                    
                    with col2:
                        st.write("🌙 **Evening Routine**")
                        if routines and len(routines) > 1:
                            evening = routines[1].replace("Evening:", "").strip()
                            for step in evening.split('\n'):
                                if step.strip():  # Only display non-empty steps
                                    st.write(f"- {step.strip()}")
                        else:
                            st.write("- Double cleanse\n- Treatment products\n- Night moisturizer")
                    
                    # Product Recommendations
                    st.write("\n**Recommended Products for Your Skin**")
                    st.write("These products are specially chosen for your skin type and concerns:")
                    app.display_product_recommendations(recommendations['products'], 
                                                     answers['product_preferences']['budget_range'])
                    
                    # Natural Care Tips
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("🌿 **Natural Care Tips**")
                        for remedy in recommendations['home_remedies']:
                            st.write(f"- {remedy}")
                    
                    with col2:
                        st.write("💫 **Lifestyle Tips for Healthy Skin**")
                        for tip in recommendations['lifestyle_tips']:
                            st.write(f"- {tip}")
                    
                    # Additional Tips
                    st.write("\n**Remember:**")
                    st.write("""
                    - Consistency is key in skincare
                    - Patch test new products before full application
                    - Listen to your skin and adjust routine as needed
                    - Protection from sun damage is essential
                    """)
    
    elif page == "About":
        st.write("""
        ### About This System
        
        This AI-powered skin analysis system is specifically designed for Indian skin types. 
        It combines advanced machine learning with traditional skincare knowledge to provide 
        personalized recommendations that are:
        
        - ✨ Suitable for Indian skin
        - 💰 Budget-friendly
        - 🌿 Considers local climate
        - 🏥 Evidence-based
        """)
    
    else:  # How It Works
        st.write("""
        ### How It Works
        
        1. **Upload Photo**: Provide a clear, well-lit photo of your face
        2. **Answer Questions**: Complete the skin assessment
        3. **AI Analysis**: Our system analyzes your skin type
        4. **Custom Recommendations**: Get personalized product and routine suggestions
        
        For best results:
        - Use a recent photo
        - Ensure good lighting
        - Remove makeup before taking the photo
        - Answer questions honestly
        """)

if __name__ == "__main__":
    main() 