import nlp_engine as nlp
from scripts.reporting import generate_pdf_report
import os

def test_advanced():
    print("--- Advanced Features Verification ---")
    
    # 1. Test LinkedIn Detection
    li_text = """
    Contact
    www.linkedin.com/in/johndoe (LinkedIn)
    Top Skills
    Python
    Machine Learning
    Summary
    Experience
    Senior Data Scientist
    Education
    """
    is_li = nlp.is_linkedin_pdf(li_text)
    print(f"LinkedIn Detected: {is_li}")
    
    # 2. Test PDF Generation (Dry Run)
    target_skills = ["Python", "Machine Learning"]
    results = [{
        "Candidate": "LinkedIn_John_Doe.pdf",
        "Rank Score": 85.5,
        "Predicted Role": "DATA-SCIENCE",
        "ML Conf %": 92.0,
        "Skills": ["Python", "Machine Learning"],
        "Gaps": ["SQL"]
    }]
    
    try:
        pdf_bytes = generate_pdf_report("Weneed a scientist", target_skills, results)
        print(f"PDF Generated Successfully ({len(pdf_bytes)} bytes)")
    except Exception as e:
        print(f"PDF Generation Failed: {e}")

if __name__ == "__main__":
    test_advanced()
