import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import json

class SkinChatBot:
    def __init__(self, root):
        self.root = root
        self.root.title("Skin Analysis Chatbot")
        
        # Load model and recommendations
        self.model = tf.keras.models.load_model('skin_classifier_model.h5')
        with open('product_recommendations.json', 'r') as f:
            self.recommendations = json.load(f)
        
        # Initialize state variables
        self.current_question = 0
        self.answers = {
            "basic_info": {},
            "skin_characteristics": {},
            "skin_concerns": {},
            "lifestyle_factors": {},
            "product_preferences": {}
        }
        self.uploaded_image = None
        
        # Create GUI elements
        self.create_gui()
        
        # Start the conversation
        self.send_bot_message("👋 Hi! I'm your skin analysis assistant. Let's start by analyzing a photo of your face. Please upload a clear, well-lit image.")
        
    def create_gui(self):
        # Create main chat frame
        self.chat_frame = ttk.Frame(self.root)
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create chat display
        self.chat_display = tk.Text(self.chat_frame, wrap=tk.WORD, width=50, height=20)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.chat_display.config(state=tk.DISABLED)
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(self.chat_frame, command=self.chat_display.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_display.config(yscrollcommand=scrollbar.set)
        
        # Create input frame
        input_frame = ttk.Frame(self.root)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create input field
        self.input_field = ttk.Entry(input_frame)
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Create send button
        send_button = ttk.Button(input_frame, text="Send", command=self.handle_input)
        send_button.pack(side=tk.RIGHT)
        
        # Create upload button
        self.upload_button = ttk.Button(input_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(side=tk.RIGHT, padx=5)
        
        # Bind Enter key to handle_input
        self.input_field.bind("<Return>", lambda e: self.handle_input())
        
    def send_bot_message(self, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "🤖 Bot: " + message + "\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
    def send_user_message(self, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "👤 You: " + message + "\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            is_valid, message = self.validate_face_image(file_path)
            if is_valid:
                self.uploaded_image = file_path
                self.send_user_message("Image uploaded successfully!")
                self.send_bot_message("Great! I can see your face clearly in the image. Now, let me ask you a few questions to better understand your skin.")
                self.ask_next_question()
            else:
                self.send_bot_message(f"⚠️ {message} Please try uploading a different photo.")

    def ask_next_question(self):
        # Define questions sequence
        questions = [
            ("basic_info", "age", "What is your age?"),
            ("basic_info", "gender", "What is your gender? (Female/Male/Other)"),
            ("basic_info", "climate", "What climate do you live in?\n1. Tropical/Humid\n2. Hot and Dry\n3. Moderate\n4. Cold and Dry\n5. Coastal/Humid"),
            ("skin_characteristics", "skin_feel", "How does your skin feel after washing?\n1. Tight and uncomfortable\n2. Slightly dry but comfortable\n3. Normal and comfortable\n4. Slightly oily in T-zone\n5. Oily all over"),
            ("skin_characteristics", "morning_skin", "How does your skin look in the morning?\n1. Dry and flaky\n2. Normal\n3. Slightly oily\n4. Very oily\n5. Combination"),
            ("skin_concerns", "primary_concern", "What is your primary skin concern?\n1. Dryness and flaking\n2. Oiliness and shine\n3. Acne and breakouts\n4. Aging/Fine lines\n5. None"),
            ("lifestyle_factors", "water_intake", "How many glasses of water do you drink daily?\n1. Less than 4\n2. 4-6\n3. 6-8\n4. More than 8"),
            ("product_preferences", "budget_range", "What's your monthly skincare budget (₹)?\n1. 500-1000\n2. 1000-2000\n3. 2000-3000\n4. 3000+")
        ]
        
        if self.current_question < len(questions):
            category, key, question = questions[self.current_question]
            self.current_category = category
            self.current_key = key
            self.send_bot_message(question)
        else:
            self.analyze_results()

    def handle_input(self):
        user_input = self.input_field.get()
        if user_input:
            self.send_user_message(user_input)
            self.input_field.delete(0, tk.END)
            
            # Store answer
            self.answers[self.current_category][self.current_key] = user_input
            
            # Move to next question
            self.current_question += 1
            self.ask_next_question()

    def analyze_results(self):
        # Analyze image and questionnaire
        skin_type, confidence, scores = self.analyze_skin_type(self.uploaded_image, self.answers)
        
        # Send analysis results
        self.send_bot_message(f"Based on your photo and responses, your skin type appears to be {skin_type.capitalize()}! 🌟")
        
        # Send personalized description
        skin_descriptions = {
            'dry': "Your skin tends to feel tight and needs extra moisture. This is common and can be effectively managed with the right skincare routine.",
            'oily': "Your skin produces more natural oils, which can actually be beneficial as it helps maintain youthful skin!",
            'normal': "You have a well-balanced skin type! While your skin is generally stable, it's still important to maintain a consistent skincare routine."
        }
        self.send_bot_message(skin_descriptions[skin_type])
        
        # Send recommendations
        self.send_recommendations(skin_type)

    def validate_face_image(self, image_path):
        """
        Validate that the uploaded image contains a clearly visible face
        Returns: (is_valid, message)
        """
        try:
            # Convert PIL Image to CV2 format
            img = Image.open(image_path)
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
            return False, f"Error processing image. Please try a different photo. Error: {str(e)}"

    def analyze_skin_type(self, image_path, answers):
        """Analyze skin type based on image and questionnaire"""
        # Get model prediction
        img = Image.open(image_path)
        img = img.resize((224, 224))
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

    def send_recommendations(self, skin_type):
        # Send routine
        routine = self.recommendations[skin_type]['routine']
        self.send_bot_message("Here's your recommended skincare routine:\n\n" + routine)
        
        # Send product recommendations
        self.send_bot_message("\nRecommended products for your skin type:")
        for product in self.recommendations[skin_type]['products']:
            self.send_bot_message(f"• {product['name']} (₹{product['price']})")
        
        # Send home remedies
        self.send_bot_message("\nNatural care tips:")
        for remedy in self.recommendations[skin_type]['home_remedies']:
            self.send_bot_message(f"• {remedy}")

def main():
    root = tk.Tk()
    app = SkinChatBot(root)
    root.mainloop()

if __name__ == "__main__":
    main() 