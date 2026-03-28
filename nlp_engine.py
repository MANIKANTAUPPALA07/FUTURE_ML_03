import os
import re
import PyPDF2
import docx
import spacy
from spacy.matcher import PhraseMatcher
import joblib
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

# Load models and NLP
nlp = spacy.load('en_core_web_sm')

# Global cache
_classifier = None
_vectorizer = None
_label_encoder = None
_skill_bank = None

def load_resources():
    global _classifier, _vectorizer, _label_encoder, _skill_bank
    if _classifier is None:
        try:
            # Fix: Use absolute path relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, 'models')
            
            _classifier = joblib.load(os.path.join(models_dir, 'resume_classifier_v2.pkl'))
            _vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer_v2.pkl'))
            _label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder_v2.pkl'))
            
            skills_path = os.path.join(models_dir, 'skills.json')
            with open(skills_path, 'r') as f:
                _skill_bank = json.load(f)
                
            print(f"Resources loaded successfully from {models_dir}")
        except Exception as e:
            print(f"Error loading resources: {e}")
            _skill_bank = []
            raise RuntimeError(f"Engine resources failed to load: {e}")
    return _classifier, _vectorizer, _label_encoder, _skill_bank

def clean_text(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', ' ', text)
    text = re.sub('@\S+', ' ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]',r' ', text) 
    text = re.sub('\s+', ' ', text)
    return text.lower().strip()

def extract_text_universal(file_obj, filename):
    ext = os.path.splitext(filename)[1].lower()
    text = ""
    try:
        if ext == '.txt':
            text = file_obj.read().decode('utf-8')
        elif ext == '.pdf':
            reader = PyPDF2.PdfReader(file_obj)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif ext == '.docx':
            doc = docx.Document(file_obj)
            text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading {filename}: {e}")
    return text

def get_skills_matcher(skills_list):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(text) for text in skills_list]
    matcher.add("SkillsMatcher", patterns)
    return matcher

def extract_skills(text, matcher):
    doc = nlp(text)
    matches = matcher(doc)
    found = set()
    for match_id, start, end in matches:
        found.add(doc[start:end].text.lower().title())
    return list(found)

def is_linkedin_pdf(text):
    # LinkedIn PDFs often contain specific patterns
    patterns = [
        r"Contact\s+www\.linkedin\.com",
        r"Top\s+Skills",
        r"Languages",
        r"Certifications",
        r"Summary",
        r"Experience",
        r"Education"
    ]
    matches = 0
    for p in patterns:
        if re.search(p, text): matches += 1
    return matches >= 3

def extract_linkedin_sections(text):
    sections = {
        "Summary": "",
        "Experience": "",
        "Skills": ""
    }
    
    # LinkedIn PDF structure is often hierarchical
    # Let's try to extract Skills specifically
    skill_match = re.search(r"Top\s+Skills\s+(.*?)(?:\n\n|\n[A-Z]|$)", text, re.DOTALL)
    if skill_match:
        sections["Skills"] = skill_match.group(1).strip()
        
    exp_match = re.search(r"Experience\s+(.*?)(?:Education|Certifications|$)", text, re.DOTALL)
    if exp_match:
        sections["Experience"] = exp_match.group(1).strip()
        
    return sections

# Helper function to load ML models specifically for prediction
def load_ml_models():
    clf, vect, le, _ = load_resources()
    return clf, vect, le

# Helper function for cleaning text for ML prediction
def clean_for_ml(text):
    return clean_text(text)

def predict_category(text):
    # If it's LinkedIn, maybe we can give it more weight in parsing
    is_li = is_linkedin_pdf(text)
    if is_li:
        print("LinkedIn profile detected.")
        
    # Try ML prediction
    clf, vect, le = load_ml_models()
    if clf and vect and le:
        clean_txt = clean_for_ml(text)
        features = vect.transform([clean_txt])
        prediction = clf.predict(features)[0]
        # Get probability/confidence
        probs = clf.predict_proba(features)[0]
        confidence = round(np.max(probs) * 100, 2)
        return le.inverse_transform([prediction])[0], confidence
    return "Unknown", 0.0 # Default return if models not loaded

def get_ml_prediction(text):
    # This function now calls the new predict_category for consistency
    return predict_category(text)

def calculate_match_score(jd_text, resume_text, jd_skills, resume_skills):
    # 1. Semantic Similarity (50%)
    # Use the trained vectorizer for consistency
    _, vect, _, _ = load_resources()
    jd_cleaned = clean_text(jd_text)
    res_cleaned = clean_text(resume_text)
    
    jd_vec = vect.transform([jd_cleaned])
    res_vec = vect.transform([res_cleaned])
    
    sim = cosine_similarity(jd_vec, res_vec).flatten()[0]
    
    # 2. Skill Match (50%)
    if not jd_skills:
        skill_score = 1.0
    else:
        matched = set(jd_skills) & set(resume_skills)
        skill_score = len(matched) / len(jd_skills)
        
    final_score = (sim * 50) + (skill_score * 50)
    return round(final_score, 2), round(sim * 100, 2), round(skill_score * 100, 2)

def identify_gap(jd_skills, resume_skills):
    gap = list(set(jd_skills) - set(resume_skills))
    return gap
