import nlp_engine as nlp
import os
import pandas as pd
from scripts.reporting import generate_pdf_report
from io import BytesIO

def run_demo():
    print("="*60)
    print("PROSCREEN AI - ML SCREENING DEMO RESULTS")
    print("="*60)
    
    # 1. Load Resources
    nlp.load_resources()
    
    # 2. Sample Job Description
    jd_txt = "We are seeking a Senior Accountant with expertise in financial reporting, tax preparation, and QuickBooks. Experience with auditing and general ledger management is a plus."
    target_skills = ["Accounting", "Financial Reporting", "Tax Preparation", "Quickbooks", "Auditing"]
    
    print(f"\n[JOB DESCRIPTION]: {jd_txt}")
    print(f"[TARGET SKILLS]: {', '.join(target_skills)}\n")
    
    # 3. Process Sample Resumes from data/
    data_dir = "data"
    resumes = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')][:5] # Top 5 for demo
    
    results = []
    jd_matcher = nlp.get_skills_matcher(target_skills)
    
    print("Processing resumes...")
    for filename in resumes:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'rb') as f:
            content = nlp.extract_text_universal(f, filename)
            
            if not content: continue
            
            role, conf = nlp.get_ml_prediction(content)
            found_skills = nlp.extract_skills(content, jd_matcher)
            score, semantic, skill_pct = nlp.calculate_match_score(jd_txt, content, target_skills, found_skills)
            gaps = nlp.identify_gap(target_skills, found_skills)
            
            results.append({
                "Candidate": filename,
                "Match%": score,
                "Role": role,
                "Conf%": conf,
                "Sim%": semantic,
                "Skills": ", ".join(found_skills),
                "Missing": ", ".join(gaps)
            })

    # 4. Display Results Table
    df = pd.DataFrame(results).sort_values("Match%", ascending=False)
    print("\n" + df.to_string(index=False))
    
    # 5. Generate PDF Report
    print("\nGenerating Comprehensive PDF Report...")
    try:
        report_data = []
        for r in results:
             report_data.append({
                "Candidate": r['Candidate'],
                "Rank Score": r['Match%'],
                "Predicted Role": r['Role'],
                "ML Conf %": r['Conf%'],
                "Skills": r['Skills'].split(", ") if r['Skills'] else [],
                "Gaps": r['Missing'].split(", ") if r['Missing'] else []
            })
        
        pdf_bytes = generate_pdf_report(jd_txt, target_skills, report_data)
        report_path = "Demo_Screening_Report.pdf"
        with open(report_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"Success! Report saved as: {os.path.abspath(report_path)}")
    except Exception as e:
        print(f"Report Generation Failed: {e}")

    print("\n" + "="*60)
    print("DEMO COMPLETE. Run 'streamlit run app.py' to see the full UI.")
    print("="*60)

if __name__ == "__main__":
    run_demo()
