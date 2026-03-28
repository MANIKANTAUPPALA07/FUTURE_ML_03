import nlp_engine as nlp
from io import BytesIO
import os

def test_repro():
    print("Testing reproduction...")
    
    # 1. Mock a simple JD and Resume
    jd_txt = "xyz123abc nonsensical" # Likely not in TF-IDF vocab
    resume_content = b"I am a Python developer with SQL skills." # Bytes as if from uploader
    
    # Simulate app.py flow
    _, _, _, base_skills = nlp.load_resources()
    print(f"Base skills: {base_skills}")
    
    temp_matcher = nlp.get_skills_matcher(base_skills)
    inferred_jd_skills = nlp.extract_skills(jd_txt, temp_matcher)
    print(f"Inferred JD skills: {inferred_jd_skills}")
    
    target_skills = inferred_jd_skills
    
    # Simulate Execute button
    jd_matcher = nlp.get_skills_matcher(target_skills if target_skills else base_skills)
    
    f_bytes = BytesIO(resume_content)
    content = nlp.extract_text_universal(f_bytes, "test.txt")
    print(f"Extracted content: '{content}'")
    
    if not content:
        print("Error: Content is empty!")
        return

    # ML logic
    pred_role, conf = nlp.get_ml_prediction(content)
    print(f"Prediction: {pred_role} ({conf}%)")
    
    res_skills = nlp.extract_skills(content, jd_matcher)
    print(f"Extracted res skills: {res_skills}")
    
    score, semantic_sim, skill_pct = nlp.calculate_match_score(jd_txt, content, target_skills, res_skills)
    print(f"Score: {score}, Sim: {semantic_sim}, Skill: {skill_pct}")
    
    print("Reproduction test complete.")

if __name__ == "__main__":
    test_repro()
