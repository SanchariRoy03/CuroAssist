"""
Enhanced Multilingual Health Chatbot with Alert System - FIXED VERSION
Features: User Authentication, MongoDB Storage, Email Alerts
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
import re
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import warnings
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import os
from functools import wraps
import secrets

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Handle MongoDB operations with enhanced debugging."""
    
    def __init__(self, connection_string="mongodb://localhost:27017/", db_name="health_chatbot"):
        """Initialize MongoDB connection."""
        try:
            self.client = MongoClient(connection_string)
            self.db = self.client[db_name]
            self.users = self.db.users
            self.health_records = self.db.health_records
            self.doctors = self.db.doctors
            self.alert_logs = self.db.alert_logs
            
            # Test the connection
            self.client.admin.command('ping')
            print(f"MongoDB connected successfully to {db_name}")
            
            # Initialize sample doctors data if not exists
            self._initialize_doctors_data()
            
            logger.info("MongoDB connection established successfully")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            print(f"MongoDB connection failed: {e}")
            self.client = None
    
    def _initialize_doctors_data(self):
        """Initialize sample doctors data if collection is empty."""
        try:
            if self.doctors.count_documents({}) == 0:
                sample_doctors = [
                    {
                        "name": "Dr. Rajesh Kumar",
                        "specialization": "General Medicine",
                        "email": "dr.rajesh@hospital.com",
                        "phone": "+91-9876543210",
                        "hospital": "City General Hospital",
                        "address": "123 Medical Center, Delhi",
                        "emergency_available": True
                    },
                    {
                        "name": "Dr. Priya Sharma",
                        "specialization": "Emergency Medicine",
                        "email": "dr.priya@emergency.com",
                        "phone": "+91-9876543211",
                        "hospital": "Emergency Care Center",
                        "address": "456 Emergency Lane, Mumbai",
                        "emergency_available": True
                    },
                    {
                        "name": "Dr. Amit Singh",
                        "specialization": "Cardiology",
                        "email": "dr.amit@cardio.com",
                        "phone": "+91-9876543212",
                        "hospital": "Heart Care Institute",
                        "address": "789 Heart Street, Bangalore",
                        "emergency_available": True
                    }
                ]
                self.doctors.insert_many(sample_doctors)
                logger.info("Sample doctors data initialized")
        except Exception as e:
            logger.error(f"Error initializing doctors data: {e}")
    
    def user_exists(self, email: str) -> bool:
        """Check if user already exists."""
        try:
            if not self.client:
                print("Database not connected")
                return False
            user = self.users.find_one({"email": email})
            exists = user is not None
            print(f"User exists check for {email}: {exists}")
            return exists
        except Exception as e:
            print(f"Error checking user existence: {e}")
            logger.error(f"Error checking user existence: {e}")
            return False
    
    def create_user(self, name: str, email: str, password: str) -> bool:
        """Create a new user account with detailed logging."""
        try:
            print(f"Creating user: {email}")
            
            if not self.client:
                print("Database not connected")
                return False
                
            # Check if user already exists
            if self.user_exists(email):
                print(f"User {email} already exists")
                return False
            
            # Hash the password with explicit method
            print(f"Hashing password for {email}")
            password_hash = generate_password_hash(password, method='pbkdf2:sha256')
            print(f"Password hash generated (first 50 chars): {password_hash[:50]}...")
            
            user_data = {
                "name": name,
                "email": email,
                "password": password_hash,
                "created_at": datetime.now(),
                "last_login": datetime.now()
            }
            
            print(f"Inserting user data into database...")
            result = self.users.insert_one(user_data)
            
            if result.inserted_id:
                print(f"User created successfully: {email} (ID: {result.inserted_id})")
                logger.info(f"User created successfully: {email}")
                return True
            else:
                print("Failed to insert user into database")
                return False
                
        except Exception as e:
            print(f"Error creating user: {e}")
            logger.error(f"Error creating user: {e}")
            return False
    
    def verify_user(self, email: str, password: str):
        """Verify user credentials with comprehensive debugging."""
        print(f"\n=== LOGIN ATTEMPT DEBUG ===")
        print(f"Email: {email}")
        print(f"Password length: {len(password) if password else 0}")
        print(f"Password: {password}")  # REMOVE THIS IN PRODUCTION!
        
        try:
            if not self.client:
                print("ERROR: Database not connected")
                return None
                
            if not email or not password:
                print("ERROR: Email or password is empty")
                return None
            
            print(f"Looking up user in database: {email}")
            # Find user by email
            user = self.users.find_one({"email": email})
            
            if not user:
                print(f"ERROR: No user found with email: {email}")
                # Show what users actually exist (for debugging)
                user_count = self.users.count_documents({})
                print(f"Total users in database: {user_count}")
                if user_count > 0:
                    print("Existing users:")
                    for existing_user in self.users.find({}, {"email": 1, "name": 1}).limit(5):
                        print(f"  - {existing_user.get('email', 'No email')}")
                return None
            
            print(f"SUCCESS: Found user: {user.get('name', 'No name')}")
            print(f"Email match: {user.get('email') == email}")
            
            stored_hash = user.get("password", "")
            if not stored_hash:
                print("ERROR: No password hash found for user")
                return None
            
            print(f"Password hash exists: {bool(stored_hash)}")
            print(f"Password hash length: {len(stored_hash)}")
            print(f"Password hash (first 50 chars): {stored_hash[:50]}...")
            
            print(f"Testing password verification...")
            # Test password verification
            verification_result = check_password_hash(stored_hash, password)
            print(f"Password verification result: {verification_result}")
            
            if verification_result:
                print("SUCCESS: Password verification successful")
                
                # Update last login
                try:
                    update_result = self.users.update_one(
                        {"_id": user["_id"]}, 
                        {"$set": {"last_login": datetime.datetime.now(datetime.timezone.utc)}}
                    )
                    print(f"Last login updated: {update_result.modified_count} documents")
                except Exception as update_error:
                    print(f"Warning: Failed to update last login: {update_error}")
                
                print("=== LOGIN SUCCESS ===\n")
                return user
            else:
                print("ERROR: Password verification failed")
                
                # Additional debugging - test hash generation
                print("Additional debugging:")
                test_hash = generate_password_hash(password, method='pbkdf2:sha256')
                test_verify = check_password_hash(test_hash, password)
                print(f"Self-test (generate new hash and verify): {test_verify}")
                
                # Check hash format
                hash_parts = stored_hash.split('$')
                print(f"Stored hash format parts: {len(hash_parts)}")
                if len(hash_parts) > 0:
                    print(f"Hash method: {hash_parts[0]}")
                
                print("=== LOGIN FAILED ===\n")
                return None
                
        except Exception as e:
            print(f"EXCEPTION in verify_user: {e}")
            logger.error(f"Error in verify_user: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_health_record(self, user_id: str, record_data: dict) -> bool:
        """Save health consultation record."""
        try:
            if not self.client:
                return False
                
            record_data["user_id"] = user_id
            record_data["timestamp"] = datetime.datetime.now(datetime.timezone.utc)
            result = self.health_records.insert_one(record_data)
            return bool(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving health record: {e}")
            return False
    
    def get_emergency_doctors(self) -> list:
        """Get list of emergency available doctors."""
        try:
            if not self.client:
                return []
            return list(self.doctors.find({"emergency_available": True}))
        except Exception as e:
            logger.error(f"Error getting emergency doctors: {e}")
            return []
    
    def log_alert(self, alert_data: dict) -> bool:
        """Log alert information."""
        try:
            if not self.client:
                return False
                
            alert_data["timestamp"] = datetime.datetime.now(datetime.timezone.utc)
            result = self.alert_logs.insert_one(alert_data)
            return bool(result.inserted_id)
        except Exception as e:
            logger.error(f"Error logging alert: {e}")
            return False
        
class EmailAlertSystem:
    """Handle email alerts for critical health conditions."""
    
    def __init__(self, smtp_server="smtp.gmail.com", smtp_port=587):
        """Initialize email system."""
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        # These should be environment variables in production
        self.sender_email = os.getenv("ALERT_EMAIL", "healthbot@example.com")
        self.sender_password = os.getenv("ALERT_EMAIL_PASSWORD", "your_app_password")
        
    def send_emergency_alert(self, user_data: Dict, condition: str, doctor: Dict) -> bool:
        """Send emergency alert email to user with doctor details."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = user_data.get('email', '')
            msg['Subject'] = "URGENT: Health Alert - Immediate Medical Attention Required"
            
            # Email body
            body = f"""
            HEALTH EMERGENCY ALERT
            =====================
            
            Dear {user_data.get('name', 'User')},
            
            Based on your reported symptoms, our system has detected a potentially serious condition: {condition}
            
            WARNING: IMMEDIATE ACTION REQUIRED
            
            Please contact the following doctor immediately or visit the nearest emergency room:
            
            RECOMMENDED DOCTOR:
            ------------------
            Name: {doctor.get('name', 'N/A')}
            Specialization: {doctor.get('specialization', 'N/A')}
            Hospital: {doctor.get('hospital', 'N/A')}
            Phone: {doctor.get('phone', 'N/A')}
            Email: {doctor.get('email', 'N/A')}
            Address: {doctor.get('address', 'N/A')}
            
            IMPORTANT NOTES:
            ---------------
            • This is an automated alert based on symptom analysis
            • Do not delay seeking professional medical help
            • If symptoms worsen, call emergency services immediately
            • Keep this email for reference during your medical consultation
            
            Emergency Contacts:
            • Ambulance: 102 (India)
            • Emergency Services: 112 (India)
            
            Time of Alert: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Disclaimer: This alert is based on automated analysis and should not replace professional medical judgment. Always consult qualified healthcare professionals for medical decisions.
            
            Stay Safe,
            CuroAssist Health Alert System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            text = msg.as_string()
            server.sendmail(self.sender_email, user_data.get('email', ''), text)
            server.quit()
            
            logger.info(f"Emergency alert sent to {user_data.get('email', '')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send emergency alert: {e}")
            return False

class MultilingualHealthChatbot:
    """Enhanced chatbot with alert system and user management."""

    def __init__(self, db_manager: DatabaseManager, email_system: EmailAlertSystem):
        """Initialize the chatbot with database and email systems."""
        self.db_manager = db_manager
        self.email_system = email_system
        self.model = None
        self.symptoms_dict = {}
        self.diseases_list = {}
        self.datasets = {}
        self.last_prediction = None
        self.last_data = {}
        self.user_language = 'en'
        self.current_user = None

        # Initialize NLP components
        self.stemmer = PorterStemmer()

        # Critical conditions that require immediate attention
        self.critical_conditions = {
            'Heart attack', 'Paralysis (brain hemorrhage)', 'Hepatitis A',
            'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
            'Alcoholic hepatitis', 'Chronic cholestasis', 'Tuberculosis',
            'Pneumonia', 'Malaria', 'Dengue', 'Typhoid', 'AIDS'
        }

        # Supported Indian languages
        self.supported_languages = {
            'hi': 'Hindi', 'bn': 'Bengali', 'te': 'Telugu', 'mr': 'Marathi',
            'ta': 'Tamil', 'gu': 'Gujarati', 'kn': 'Kannada', 'ml': 'Malayalam',
            'pa': 'Punjabi', 'or': 'Odia', 'as': 'Assamese', 'ur': 'Urdu', 'en': 'English'
        }

        # Common symptom translations
        self.symptom_translations = {
            # Hindi
            'बुखार': 'fever', 'सिरदर्द': 'headache', 'खांसी': 'cough',
            'उल्टी': 'vomiting', 'दस्त': 'diarrhea', 'पेट दर्द': 'stomach pain',
            'सांस लेने में तकलीफ': 'breathlessness', 'कमजोरी': 'weakness',
            'चक्कर आना': 'dizziness', 'जोड़ों का दर्द': 'joint pain',
            # Bengali
            'জ্বর': 'fever', 'মাথাব্যথা': 'headache', 'কাশি': 'cough',
            'বমি': 'vomiting', 'পেটের ব্যথা': 'stomach pain',
            'শ্বাসকষ্ট': 'breathlessness', 'দুর্বলতা': 'weakness',
            # Tamil
            'காய்ச்சல்': 'fever', 'தலைவலி': 'headache', 'இருமல்': 'cough',
            'வாந்தி': 'vomiting', 'வயிற்று வலி': 'stomach pain',
            'மூச்சுத் திணறல்': 'breathlessness', 'பலவீனம்': 'weakness',
            # Telugu
            'జ్వరం': 'fever', 'తలనొప్పి': 'headache', 'దగ్గు': 'cough',
            'వాంతులు': 'vomiting', 'కడుపు నొప్పి': 'stomach pain',
            'ఊపిరాడకపోవడం': 'breathlessness', 'బలహీనత': 'weakness',
        }

        # Initialize components
        self._initialize_components()

    def set_current_user(self, user_data: Dict):
        """Set current user for the session."""
        self.current_user = user_data

    def _initialize_components(self):
        """Initialize all components with error handling."""
        try:
            self._load_model()
            self._load_symptoms_mapping()
            self._load_diseases_mapping()
            self._load_datasets()
            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            self._set_fallback_data()

    def _set_fallback_data(self):
        """Set fallback data when files are not available."""
        logger.warning("Using fallback data...")
        
        self.symptoms_dict = {
            'fever': 0, 'headache': 1, 'cough': 2, 'vomiting': 3,
            'stomach_pain': 4, 'diarrhea': 5, 'weakness': 6, 'dizziness': 7,
            'chest_pain': 8, 'breathlessness': 9, 'high_fever': 10
        }
        
        self.diseases_list = {
            0: 'Common Cold', 1: 'Flu', 2: 'Gastroenteritis',
            3: 'Heart attack', 4: 'Pneumonia', 5: 'Tuberculosis'
        }
        
        self.datasets = {
            'description': pd.DataFrame(columns=['Disease', 'Description']),
            'precautions': pd.DataFrame(columns=['Disease']),
            'medications': pd.DataFrame(columns=['Disease', 'Medication']),
            'diets': pd.DataFrame(columns=['Disease', 'Diet']),
            'workout': pd.DataFrame(columns=['disease', 'workout'])
        }

    def detect_language(self, text: str) -> str:
        """Detect the language of input text."""
        try:
            clean_text = re.sub(r'[^\w\s\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]', ' ', text)
            
            if len(clean_text.strip()) < 3:
                return 'en'
                
            detected_lang = detect(clean_text)
            if detected_lang in self.supported_languages:
                return detected_lang
            return 'en'
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return 'en'

    def translate_text(self, text: str, target_lang: str = 'en', source_lang: Optional[str] = None) -> str:
        """Translate text between languages."""
        try:
            if not text.strip() or source_lang == target_lang:
                return text
                
            if source_lang:
                translator = GoogleTranslator(source=source_lang, target=target_lang)
            else:
                translator = GoogleTranslator(source='auto', target=target_lang)
                
            result = translator.translate(text)
            return result if result else text
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text

    def preprocess_multilingual_input(self, user_input: str) -> Tuple[str, str]:
        """Preprocess multilingual input and return English translation."""
        if not user_input.strip():
            return "", "en"
            
        detected_lang = self.detect_language(user_input)
        self.user_language = detected_lang

        cleaned_input = re.sub(
            r'[^\w\s\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0E00-\u0E7F]',
            ' ', user_input
        )
        cleaned_input = ' '.join(cleaned_input.split())

        english_symptoms = []
        words = cleaned_input.split()
        
        # Check for multi-word combinations and single words
        for i in range(len(words) - 2):
            combo = ' '.join(words[i:i+3])
            if combo.lower() in self.symptom_translations:
                english_symptoms.append(self.symptom_translations[combo.lower()])
                
        for i in range(len(words) - 1):
            combo = ' '.join(words[i:i+2])
            if combo.lower() in self.symptom_translations:
                english_symptoms.append(self.symptom_translations[combo.lower()])
                
        for word in words:
            if word.lower() in self.symptom_translations:
                english_symptoms.append(self.symptom_translations[word.lower()])

        if english_symptoms:
            english_text = ', '.join(set(english_symptoms))
        elif detected_lang != 'en':
            english_text = self.translate_text(cleaned_input, 'en', detected_lang)
        else:
            english_text = cleaned_input

        return english_text.lower(), detected_lang

    def extract_symptoms_nlp(self, text: str) -> List[str]:
        """Extract symptoms using NLP techniques."""
        if not text.strip():
            return []
            
        try:
            tokens = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
        except:
            filtered_tokens = text.lower().split()

        matched_symptoms = []
        text_lower = text.lower()

        # Direct symptom matching
        for symptom in self.symptoms_dict.keys():
            symptom_words = symptom.replace('_', ' ')
            if symptom_words in text_lower or symptom in text_lower:
                matched_symptoms.append(symptom)

        # Enhanced fuzzy matching
        symptom_variants = {
            'fever': ['high_fever', 'mild_fever', 'temperature'],
            'pain': ['joint_pain', 'stomach_pain', 'back_pain', 'chest_pain', 'abdominal_pain', 'belly_pain'],
            'headache': ['headache', 'head_pain'],
            'cough': ['cough', 'coughing'],
            'vomit': ['vomiting', 'nausea', 'throwing_up'],
            'diarr': ['diarrhoea', 'loose_stool'],
            'weak': ['weakness_in_limbs', 'muscle_weakness', 'fatigue'],
            'breath': ['breathlessness', 'shortness_of_breath'],
            'dizz': ['dizziness', 'vertigo'],
            'tired': ['fatigue', 'lethargy', 'exhaustion'],
            'cold': ['continuous_sneezing', 'runny_nose', 'congestion'],
            'skin': ['skin_rash', 'itching', 'rash'],
            'weight': ['weight_loss', 'weight_gain']
        }

        for key, variants in symptom_variants.items():
            if key in text_lower:
                for variant in variants:
                    if variant in self.symptoms_dict:
                        matched_symptoms.append(variant)

        return list(set(matched_symptoms))

    def _load_model(self) -> None:
        """Load the trained SVM model."""
        try:
            model_path = Path('Models/svc.pkl')
            if not model_path.exists():
                alt_paths = ['svc.pkl', 'models/svc.pkl', 'model/svc.pkl']
                for alt_path in alt_paths:
                    if Path(alt_path).exists():
                        model_path = Path(alt_path)
                        break
                else:
                    raise FileNotFoundError("Model file not found")
                    
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
            logger.info("Model loaded successfully")
        except FileNotFoundError:
            logger.error("Model file not found - using dummy model")
            self.model = None

    def _load_symptoms_mapping(self) -> None:
        """Load symptoms to index mapping."""
        self.symptoms_dict = {
            'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2,
            'continuous_sneezing': 3, 'shivering': 4, 'chills': 5,
            'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
            'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11,
            'burning_micturition': 12, 'spotting_urination': 13,
            'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
            'cold_hands_and_feets': 17, 'mood_swings': 18,
            'weight_loss': 19, 'restlessness': 20, 'lethargy': 21,
            'patches_in_throat': 22, 'irregular_sugar_level': 23,
            'cough': 24, 'high_fever': 25, 'sunken_eyes': 26,
            'breathlessness': 27, 'sweating': 28, 'dehydration': 29,
            'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
            'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35,
            'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38,
            'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41,
            'yellow_urine': 42, 'yellowing_of_eyes': 43,
            'acute_liver_failure': 44, 'fluid_overload': 45,
            'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47,
            'malaise': 48, 'blurred_and_distorted_vision': 49,
            'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52,
            'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55,
            'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
            'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60,
            'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63,
            'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67,
            'swollen_legs': 68, 'swollen_blood_vessels': 69,
            'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71,
            'brittle_nails': 72, 'swollen_extremeties': 73,
            'excessive_hunger': 74, 'extra_marital_contacts': 75,
            'drying_and_tingling_lips': 76, 'slurred_speech': 77,
            'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80,
            'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83,
            'spinning_movements': 84, 'loss_of_balance': 85,
            'unsteadiness': 86, 'weakness_of_one_body_side': 87,
            'loss_of_smell': 88, 'bladder_discomfort': 89,
            'foul_smell_of_urine': 90, 'continuous_feel_of_urine': 91,
            'passage_of_gases': 92, 'internal_itching': 93,
            'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96,
            'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99,
            'belly_pain': 100, 'abnormal_menstruation': 101,
            'dischromic_patches': 102, 'watering_from_eyes': 103,
            'increased_appetite': 104, 'polyuria': 105, 'family_history': 106,
            'mucoid_sputum': 107, 'rusty_sputum': 108,
            'lack_of_concentration': 109, 'visual_disturbances': 110,
            'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112,
            'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115,
            'history_of_alcohol_consumption': 116, 'fluid_overload_1': 117,
            'blood_in_sputum': 118, 'prominent_veins_on_calf': 119,
            'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122,
            'blackheads': 123, 'scurring': 124, 'skin_peeling': 125,
            'silver_like_dusting': 126, 'small_dents_in_nails': 127,
            'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130,
            'yellow_crust_ooze': 131
        }

    def _load_diseases_mapping(self) -> None:
        """Load disease index to name mapping."""
        self.diseases_list = {
            0: 'Paroxysmal Positional Vertigo', 1: 'AIDS', 2: 'Acne',
            3: 'Alcoholic hepatitis', 4: 'Allergy', 5: 'Arthritis',
            6: 'Bronchial Asthma', 7: 'Cervical spondylosis', 8: 'Chicken pox',
            9: 'Chronic cholestasis', 10: 'Common Cold', 11: 'Dengue',
            12: 'Diabetes', 13: 'Dimorphic hemorrhoids(piles)', 14: 'Drug Reaction',
            15: 'Fungal infection', 16: 'GERD', 17: 'Gastroenteritis',
            18: 'Heart attack', 19: 'Hepatitis B', 20: 'Hepatitis C',
            21: 'Hepatitis D', 22: 'Hepatitis E', 23: 'Hypertension',
            24: 'Hyperthyroidism', 25: 'Hypoglycemia', 26: 'Hypothyroidism',
            27: 'Impetigo', 28: 'Jaundice', 29: 'Malaria', 30: 'Migraine',
            31: 'Osteoarthritis', 32: 'Paralysis (brain hemorrhage)',
            33: 'Peptic ulcer disease', 34: 'Pneumonia', 35: 'Psoriasis',
            36: 'Tuberculosis', 37: 'Typhoid', 38: 'Urinary tract infection',
            39: 'Varicose veins', 40: 'Hepatitis A'
        }

    def _load_datasets(self) -> None:
        """Load all CSV datasets with error handling."""
        dataset_files = {
            'description': 'datasets/description.csv',
            'precautions': 'datasets/precautions_df.csv',
            'medications': 'datasets/medications.csv',
            'diets': 'datasets/diets.csv',
            'workout': 'datasets/workout_df.csv'
        }

        for name, filepath in dataset_files.items():
            try:
                if Path(filepath).exists():
                    self.datasets[name] = pd.read_csv(filepath)
                else:
                    alt_paths = [
                        filepath.replace('datasets/', ''),
                        filepath.replace('datasets/', 'data/'),
                        filepath.replace('datasets/', 'csv/')
                    ]
                    
                    loaded = False
                    for alt_path in alt_paths:
                        if Path(alt_path).exists():
                            self.datasets[name] = pd.read_csv(alt_path)
                            loaded = True
                            break
                    
                    if not loaded:
                        logger.warning(f"Dataset {filepath} not found, creating empty DataFrame")
                        if name == 'description':
                            self.datasets[name] = pd.DataFrame(columns=['Disease', 'Description'])
                        elif name == 'precautions':
                            self.datasets[name] = pd.DataFrame(columns=['Disease', 'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'])
                        elif name == 'medications':
                            self.datasets[name] = pd.DataFrame(columns=['Disease', 'Medication'])
                        elif name == 'diets':
                            self.datasets[name] = pd.DataFrame(columns=['Disease', 'Diet'])
                        elif name == 'workout':
                            self.datasets[name] = pd.DataFrame(columns=['disease', 'workout'])
                        continue
                        
                logger.info(f"Loaded {name} dataset")
                
            except Exception as e:
                logger.error(f"Error loading dataset {filepath}: {e}")
                self.datasets[name] = pd.DataFrame()

    def predict_disease(self, symptoms: List[str]) -> str:
        """Predict disease based on symptoms."""
        if not symptoms:
            return "Unknown condition"
            
        if self.model is None:
            return self._fallback_prediction(symptoms)
            
        try:
            input_vector = np.zeros(len(self.symptoms_dict))
            for symptom in symptoms:
                if symptom in self.symptoms_dict:
                    input_vector[self.symptoms_dict[symptom]] = 1
            
            if np.sum(input_vector) == 0:
                return "Unknown condition"
                
            prediction_index = self.model.predict([input_vector])[0]
            return self.diseases_list.get(prediction_index, "Unknown condition")
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._fallback_prediction(symptoms)

    def _fallback_prediction(self, symptoms: List[str]) -> str:
        """Fallback prediction when model is not available."""
        symptom_disease_map = {
            'fever': 'Common Cold',
            'cough': 'Common Cold',
            'headache': 'Migraine',
            'vomiting': 'Gastroenteritis',
            'diarrhea': 'Gastroenteritis',
            'stomach_pain': 'Gastroenteritis',
            'chest_pain': 'Heart attack',
            'breathlessness': 'Pneumonia'
        }
        
        for symptom in symptoms:
            if symptom in symptom_disease_map:
                return symptom_disease_map[symptom]
        
        return "General illness"

    def get_disease_info(self, disease: str) -> Tuple[str, List[str], List[str], List[str], List[str]]:
        """Get comprehensive information about a disease."""
        try:
            # Description
            desc_df = self.datasets.get('description', pd.DataFrame())
            if not desc_df.empty and 'Disease' in desc_df.columns:
                description_rows = desc_df[desc_df['Disease'] == disease]
                if not description_rows.empty:
                    description = " ".join(description_rows['Description'].values)
                else:
                    description = "No description available."
            else:
                description = "No description available."

            # Precautions
            prec_df = self.datasets.get('precautions', pd.DataFrame())
            if not prec_df.empty and 'Disease' in prec_df.columns:
                prec_rows = prec_df[prec_df['Disease'] == disease]
                if not prec_rows.empty:
                    precautions = prec_rows.iloc[0].dropna().tolist()[1:]
                else:
                    precautions = ["No precautions available."]
            else:
                precautions = ["No precautions available."]

            # Medications
            med_df = self.datasets.get('medications', pd.DataFrame())
            if not med_df.empty and 'Disease' in med_df.columns:
                medications = med_df[med_df['Disease'] == disease]['Medication'].tolist()
                if not medications:
                    medications = ["Consult a doctor for appropriate medication."]
            else:
                medications = ["Consult a doctor for appropriate medication."]

            # Diet
            diet_df = self.datasets.get('diets', pd.DataFrame())
            if not diet_df.empty and 'Disease' in diet_df.columns:
                diet = diet_df[diet_df['Disease'] == disease]['Diet'].tolist()
                if not diet:
                    diet = ["Maintain a balanced diet."]
            else:
                diet = ["Maintain a balanced diet."]

            # Workout
            workout_df = self.datasets.get('workout', pd.DataFrame())
            if not workout_df.empty and 'disease' in workout_df.columns:
                workout = workout_df[workout_df['disease'] == disease]['workout'].tolist()
                if not workout:
                    workout = ["Light exercise as tolerated."]
            else:
                workout = ["Light exercise as tolerated."]

            return description, precautions, medications, diet, workout
            
        except Exception as e:
            logger.error(f"Error getting disease info: {e}")
            return (
                "Information unavailable.",
                ["Consult a healthcare provider."],
                ["Consult a doctor for medication."],
                ["Maintain a balanced diet."],
                ["Light exercise as tolerated."]
            )

    def check_for_emergency(self, predicted_disease: str, symptoms: List[str]) -> bool:
        """Check if the condition requires immediate medical attention."""
        # Check if predicted disease is in critical conditions
        if predicted_disease in self.critical_conditions:
            return True
        
        # Check for emergency symptoms
        emergency_symptoms = {
            'chest_pain', 'severe_chest_pain', 'breathlessness', 'difficulty_breathing',
            'loss_of_consciousness', 'coma', 'severe_bleeding', 'high_fever',
            'weakness_of_one_body_side', 'slurred_speech', 'confusion',
            'severe_abdominal_pain', 'blood_in_vomit', 'blood_in_stool'
        }
        
        # Check if any emergency symptoms are present
        for symptom in symptoms:
            if symptom in emergency_symptoms:
                return True
        
        return False

    def process_user_input(self, user_message: str) -> Dict:
        """Process multilingual user input and return comprehensive response."""
        if not user_message.strip():
            return {
                'response': "Please provide your symptoms or ask a question.",
                'is_emergency': False,
                'predicted_disease': None
            }
            
        try:
            english_text, detected_lang = self.preprocess_multilingual_input(user_message)
            user_msg = english_text.lower().strip()

            synonym_map = {
                "description": ["description", "about", "what is", "tell me about", "details", "info", "information"],
                "precautions": ["precautions", "prevention", "prevent", "avoid", "safety", "care"],
                "medications": ["medication", "medicine", "treatment", "drugs", "pills", "remedy"],
                "diets": ["diet", "food", "nutrition", "eat", "meal", "eating"],
                "workout": ["workout", "exercise", "fitness", "physical activity", "gym"]
            }

            symptoms = self.extract_symptoms_nlp(user_msg)

            if symptoms:
                predicted_disease = self.predict_disease(symptoms)
                self.last_prediction = predicted_disease
                
                # Save health record if user is logged in
                if self.current_user and self.db_manager:
                    record_data = {
                        'symptoms': symptoms,
                        'predicted_disease': predicted_disease,
                        'user_message': user_message,
                        'detected_language': detected_lang
                    }
                    self.db_manager.save_health_record(str(self.current_user['_id']), record_data)
                
                # Check for emergency
                is_emergency = self.check_for_emergency(predicted_disease, symptoms)
                
                if is_emergency:
                    response = self._handle_emergency_case(predicted_disease, symptoms)
                else:
                    response = f"Based on your symptoms ({', '.join(symptoms)}), I think you may have **{predicted_disease}**. You can ask me about medications, precautions, diet, or exercises for this condition."
                
                return {
                    'response': self._translate_response(response, detected_lang),
                    'is_emergency': is_emergency,
                    'predicted_disease': predicted_disease,
                    'detected_language': detected_lang
                }

            # Check for specific information requests
            for info_type, keywords in synonym_map.items():
                if any(keyword in user_msg for keyword in keywords):
                    if self.last_prediction:
                        response = self.get_info_by_type(info_type, self.last_prediction)
                        return {
                            'response': self._translate_response(response, detected_lang),
                            'is_emergency': False,
                            'predicted_disease': self.last_prediction
                        }
                    else:
                        response = "Please provide your symptoms first so I can help you."
                        return {
                            'response': self._translate_response(response, detected_lang),
                            'is_emergency': False,
                            'predicted_disease': None
                        }

            # Default response
            response = "I couldn't identify any symptoms. Could you please describe what you're experiencing?"
            return {
                'response': self._translate_response(response, detected_lang),
                'is_emergency': False,
                'predicted_disease': None
            }
                
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            return {
                'response': "I'm sorry, I encountered an error processing your request. Please try again.",
                'is_emergency': False,
                'predicted_disease': None
            }

    def _handle_emergency_case(self, predicted_disease: str, symptoms: List[str]) -> str:
        """Handle emergency medical conditions."""
        emergency_response = f"""
🚨 **MEDICAL EMERGENCY DETECTED** 🚨

Based on your symptoms, you may have **{predicted_disease}**, which requires immediate medical attention.

⚠️ **IMMEDIATE ACTION REQUIRED:**
• Contact emergency services immediately (Call 102 or 112 in India)
• Go to the nearest hospital emergency room
• Do not delay seeking professional medical help

**Your symptoms:** {', '.join(symptoms)}

**Important:** This is an automated assessment. Please seek immediate professional medical care.
        """
        
        try:
            if self.current_user and self.db_manager:
                doctors = self.db_manager.get_emergency_doctors()
                if doctors:
                    # Get the most appropriate doctor (first available for now)
                    selected_doctor = doctors[0]
                    
                    # Send email alert
                    alert_sent = self.email_system.send_emergency_alert(
                        self.current_user,
                        predicted_disease,
                        selected_doctor
                    )
                    
                    if alert_sent:
                        emergency_response += (
                            "\n\n📧 **Emergency alert sent to your email with doctor contact details.**"
                        )
                    
                    # Log the alert
                    self.db_manager.log_alert({
                        'user_id': str(self.current_user['_id']),
                        'predicted_disease': predicted_disease,
                        'symptoms': symptoms,
                        'doctor_contacted': selected_doctor,
                        'alert_sent': alert_sent
                    })
                else:
                    emergency_response += (
                        "\n\n❌ **Failed to send email alert. Please contact emergency services immediately.**"
                    )

        except Exception as e:
            emergency_response += f"\n\n⚠️ Error while processing emergency alert: {str(e)}"

        return emergency_response

    def get_info_by_type(self, info_type: str, disease: str) -> str:
        """Return requested info type for disease."""
        if not disease:
            return "Please provide your symptoms first so I can help you."
            
        try:
            desc, prec, meds, diet, workout = self.get_disease_info(disease)
            
            info_map = {
                "description": f"**About {disease}:**\n{desc}",
                "precautions": f"**Precautions for {disease}:**\n• " + '\n• '.join(prec),
                "medications": f"**Medications for {disease}:**\n• " + '\n• '.join(meds),
                "diets": f"**Diet recommendations for {disease}:**\n• " + '\n• '.join(diet),
                "workout": f"**Exercise recommendations for {disease}:**\n• " + '\n• '.join(workout)
            }
            
            return info_map.get(info_type, "Information not found for this request.")
            
        except Exception as e:
            logger.error(f"Error getting info by type: {e}")
            return "Sorry, I couldn't retrieve that information right now."

    def _translate_response(self, response: str, target_lang: str) -> str:
        """Translate response to user's language if not English."""
        if target_lang != 'en' and target_lang in self.supported_languages:
            try:
                translated = self.translate_text(response, target_lang, 'en')
                if translated and translated != response:
                    return translated
                return response
            except Exception as e:
                logger.error(f"Response translation error: {e}")
                return response
        return response

# ----------------------- Flask App -----------------------
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generate a secure secret key

# Initialize systems
try:
    db_manager = DatabaseManager()
    email_system = EmailAlertSystem()
    chatbot = MultilingualHealthChatbot(db_manager, email_system)
    logger.info("All systems initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize systems: {e}")
    db_manager = None
    email_system = None
    chatbot = None

def login_required(f):
    """Decorator to require login for certain routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    """Redirect to login or chatbot based on session."""
    if 'user_id' in session:
        return redirect(url_for('chatbot_interface'))
    return redirect(url_for('login'))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    """Handle user signup."""
    if request.method == "POST":
        try:
            # Handle JSON requests (API signup)
            if request.is_json:
                data = request.get_json()
                email = data.get("email", "").lower().strip()
                password = data.get("password", "").strip()
                confirm_password = data.get("confirm", "").strip()
                name = data.get("name", "").strip() or "User"

                if not email or not password:
                    return jsonify({"success": False, "message": "Email and password required"}), 400
                
                if len(password) < 6:
                    return jsonify({"success": False, "message": "Password must be at least 6 characters"}), 400
                
                if password != confirm_password:
                    return jsonify({"success": False, "message": "Passwords do not match"}), 400

                if not db_manager:
                    return jsonify({"success": False, "message": "Database unavailable"}), 500
                
                if not db_manager.client:
                    return jsonify({"success": False, "message": "Database connection failed"}), 500

                if db_manager.user_exists(email):
                    return jsonify({"success": False, "message": "Email already registered"}), 409

                success = db_manager.create_user(name, email, password)
                if success:
                    logger.info(f"User account created successfully: {email}")
                    return jsonify({"success": True, "message": "Account created successfully"}), 201
                else:
                    return jsonify({"success": False, "message": "Failed to create account"}), 500

            # Handle form submissions (signup.html)
            email = request.form.get("email", "").lower().strip()
            password = request.form.get("password", "").strip()
            confirm_password = request.form.get("confirm", "").strip()
            name = request.form.get("name", "").strip() or email.split('@')[0]

            if not email or not password:
                return render_template("signup.html", error="Email and password required")
            
            if len(password) < 6:
                return render_template("signup.html", error="Password must be at least 6 characters")
            
            if password != confirm_password:
                return render_template("signup.html", error="Passwords do not match")

            if not db_manager:
                return render_template("signup.html", error="Database unavailable")
            
            if not db_manager.client:
                return render_template("signup.html", error="Database connection failed")

            if db_manager.user_exists(email):
                return render_template("signup.html", error="Email already registered")

            success = db_manager.create_user(name, email, password)
            if success:
                logger.info(f"User account created successfully: {email}")
                user = db_manager.users.find_one({"email": email})
                if user:
                    # Save session like login
                    session['user_id'] = str(user['_id'])
                    session['user_name'] = user['name']
                    session['user_email'] = user['email']
                    if chatbot:
                        chatbot.set_current_user(user)
                return redirect(url_for("chatbot_interface"))
            else:
                return render_template("signup.html", error="Failed to create account")

        except Exception as e:
            logger.error(f"Signup error: {e}")
            if request.is_json:
                return jsonify({"success": False, "message": "Signup failed due to server error"}), 500
            else:
                return render_template("signup.html", error="Signup failed. Please try again.")

    # GET request → render signup page
    return render_template("signup.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if request.method == 'POST':
        try:
            # Handle both JSON and form data
            if request.is_json:
                data = request.get_json()
                email = data.get('email', '').lower().strip()
                password = data.get('password', '').strip()
                
                if not email or not password:
                    return jsonify({'success': False, 'message': 'Email and password required'}), 400
                
                if not db_manager:
                    return jsonify({'success': False, 'message': 'Database unavailable'}), 500
                
                if not db_manager.client:
                    return jsonify({'success': False, 'message': 'Database connection failed'}), 500
                
                user = db_manager.verify_user(email, password)
                if user:
                    session['user_id'] = str(user['_id'])
                    session['user_name'] = user['name']
                    session['user_email'] = user['email']
                    
                    # Set current user in chatbot
                    if chatbot:
                        chatbot.set_current_user(user)
                    
                    logger.info(f"User logged in successfully: {email}")
                    return jsonify({'success': True, 'message': 'Login successful'})
                else:
                    logger.warning(f"Login attempt failed for: {email}")
                    return jsonify({'success': False, 'message': 'Invalid email or password'}), 401
            else:
                # Handle form data
                email = request.form.get('email', '').lower().strip()
                password = request.form.get('password', '').strip()
                
                if not email or not password:
                    return render_template('login.html', error='Email and password required')
                
                if not db_manager:
                    return render_template('login.html', error='Database unavailable')
                
                if not db_manager.client:
                    return render_template('login.html', error='Database connection failed')
                
                user = db_manager.verify_user(email, password)
                if user:
                    session['user_id'] = str(user['_id'])
                    session['user_name'] = user['name']
                    session['user_email'] = user['email']
                    
                    # Set current user in chatbot
                    if chatbot:
                        chatbot.set_current_user(user)
                    
                    logger.info(f"User logged in successfully: {email}")
                    return redirect(url_for('chatbot_interface'))
                else:
                    logger.warning(f"Login attempt failed for: {email}")
                    return render_template('login.html', error='Invalid email or password')
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            if request.is_json:
                return jsonify({'success': False, 'message': 'Login failed due to server error'}), 500
            else:
                return render_template('login.html', error='Login failed. Please try again.')
    
    # Handle success message from signup redirect
    message = request.args.get('message')
    return render_template('login.html', message=message)

@app.route('/logout')
def logout():
    """Handle user logout."""
    session.clear()
    if chatbot:
        chatbot.set_current_user(None)
    return redirect(url_for('login'))

@app.route('/chatbot')
@login_required
def chatbot_interface():
    """Render the main chatbot interface."""
    user_name = session.get('user_name', 'User')
    return render_template('chatbot.html', user_name=user_name)

@app.route('/get_response', methods=['POST'])
@login_required
def get_response():
    """Handle chatbot responses via API."""
    try:
        if not chatbot:
            return jsonify({
                'response': 'Sorry, the chatbot is currently unavailable. Please try again later.',
                'error': 'Chatbot initialization failed'
            }), 500
            
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'response': 'Please provide a message.',
                'error': 'Invalid request format'
            }), 400
            
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({
                'response': 'Please provide a valid message.',
                'error': 'Empty message'
            }), 400
            
        # Process the message and get comprehensive response
        result = chatbot.process_user_input(user_message)
        
        return jsonify({
            'response': result['response'],
            'detected_language': chatbot.user_language,
            'last_prediction': result.get('predicted_disease'),
            'is_emergency': result.get('is_emergency', False)
        })
        
    except Exception as e:
        logger.error(f"Error in get_response: {e}")
        return jsonify({
            'response': 'I apologize, but I encountered an error processing your request. Please try again.',
            'error': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    try:
        status = {
            'status': 'healthy' if all([chatbot, db_manager, email_system]) else 'unhealthy',
            'chatbot_initialized': chatbot is not None,
            'database_connected': db_manager is not None and db_manager.client is not None,
            'email_system_ready': email_system is not None,
            'supported_languages': list(chatbot.supported_languages.keys()) if chatbot else [],
            'model_loaded': chatbot.model is not None if chatbot else False,
            'datasets_loaded': len(chatbot.datasets) if chatbot else 0
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/profile')
@login_required
def profile():
    """User profile page."""
    return render_template('profile.html')

@app.route('/history')
@login_required
def history():
    """User consultation history."""
    try:
        if db_manager:
            user_id = session['user_id']
            # This would fetch user's health records from MongoDB
            # For now, returning a placeholder
            records = []  # db_manager.get_user_health_records(user_id)
            return render_template('history.html', records=records)
        else:
            return render_template('history.html', records=[])
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return render_template('history.html', records=[])

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Add startup checks
    print("🚀 Starting Enhanced Multilingual Health Chatbot with Alert System...")
    print("=" * 70)
    
    if chatbot:
        print("✅ Chatbot initialized successfully")
        print(f"✅ Supported languages: {len(chatbot.supported_languages)}")
        print(f"✅ Symptoms in database: {len(chatbot.symptoms_dict)}")
        print(f"✅ Diseases in database: {len(chatbot.diseases_list)}")
        print(f"✅ Datasets loaded: {len(chatbot.datasets)}")
        print(f"✅ Model loaded: {'Yes' if chatbot.model else 'No (using fallback)'}")
    else:
        print("❌ Chatbot initialization failed")
    
    if db_manager:
        print("✅ Database manager initialized")
        print(f"✅ MongoDB connected: {'Yes' if db_manager.client else 'No'}")
    else:
        print("❌ Database manager initialization failed")
    
    if email_system:
        print("✅ Email alert system initialized")
    else:
        print("❌ Email alert system initialization failed")
    
    print("=" * 70)
    print("🌐 Starting Flask server...")
    print("📝 Remember to set environment variables for email alerts:")
    print("   - ALERT_EMAIL: Your email address")
    print("   - ALERT_EMAIL_PASSWORD: Your app password")
    print("💾 MongoDB should be running on localhost:27017")
    print("🔗 Available routes:")
    print("   - /login - User login")
    print("   - /signup - User registration")
    print("   - /chatbot - Main chatbot interface")
    print("   - /health - System health check")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
            