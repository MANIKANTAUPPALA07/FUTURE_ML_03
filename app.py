import streamlit as st
import pandas as pd
import os
import nlp_engine as nlp
from scripts.reporting import generate_pdf_report
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import datetime

# Page Config
st.set_page_config(page_title="Task 3: ML Resume Screening", layout="wide", page_icon="🤖")

# Deep Design System
st.markdown("""
<style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stMetric { background-color: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; }
    .stButton>button { 
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%); 
        color: white; border: none; padding: 12px; border-radius: 8px; font-weight: 600; 
    }
    .stTab { background-color: #0f172a; }
    h1, h2, h3 { color: #60a5fa !important; font-family: 'Outfit', sans-serif; }
    .stDataFrame { background-color: #1e293b; border-radius: 12px; }
    .gap-chip { background-color: #7f1d1d; color: #fecaca; padding: 4px 10px; border-radius: 20px; margin: 2px; display: inline-block; font-size: 12px; }
    .skill-chip { background-color: #064e3b; color: #d1fae5; padding: 4px 10px; border-radius: 20px; margin: 2px; display: inline-block; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Task 3: Enterprise ML Resume Screener")
st.markdown("Automate your hiring funnel with Supervised Learning, Semantic Analysis, and **LinkedIn Integration**.")

# Load resources
_, _, _, base_skills = nlp.load_resources()

# Sidebar
st.sidebar.title("⚙️ Engine Control")
min_match = st.sidebar.slider("Min Match Threshold (%)", 0, 100, 30)
top_k = st.sidebar.number_input("Display Top K", 5, 50, 10)

# Main Grid
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Job Definition")
    jd_txt = st.text_area("Job Requirements:", height=250, placeholder="Paste JD here...")
    
    if jd_txt:
        # Extract skills from JD for baseline
        temp_matcher = nlp.get_skills_matcher(base_skills)
        inferred_jd_skills = nlp.extract_skills(jd_txt, temp_matcher)
        
        target_skills = st.multiselect("Confirm Target Skills:", 
                                       options=sorted(list(set(base_skills + inferred_jd_skills))), 
                                       default=inferred_jd_skills)
    else:
        target_skills = []

with col2:
    st.subheader("📁 Resume Pipeline")
    st.markdown("Upload Resumes or **LinkedIn Profiles** (PDF/DOCX/TXT):")
    files = st.file_uploader("Drag and drop multiple files", accept_multiple_files=True)
    if files:
        st.success(f"{len(files)} files queued.")

# Execution
if st.button("🚀 Execute ML Screening"):
    if not jd_txt or not files:
        st.error("Please provide both Job Requirements and Resumes.")
    else:
        with st.spinner("Processing through ML Vector Engine..."):
            jd_matcher = nlp.get_skills_matcher(target_skills if target_skills else base_skills)
            results = []
            
            for f in files:
                f_bytes = BytesIO(f.getvalue())
                content = nlp.extract_text_universal(f_bytes, f.name)
                
                if not content: 
                    st.warning(f"Could not extract text from {f.name}. Please check file format.")
                    continue
                
                try:
                    # ML logic
                    pred_role, conf = nlp.get_ml_prediction(content)
                    res_skills = nlp.extract_skills(content, jd_matcher)
                    score, semantic_sim, skill_pct = nlp.calculate_match_score(jd_txt, content, target_skills, res_skills)
                    gaps = nlp.identify_gap(target_skills, res_skills)
                    
                    results.append({
                        "Candidate": f.name,
                        "Rank Score": score,
                        "Predicted Role": pred_role,
                        "ML Conf %": conf,
                        "Semantic Sim %": semantic_sim,
                        "Skill Match %": skill_pct,
                        "Skills": res_skills,
                        "Gaps": gaps
                    })
                except Exception as e:
                    st.error(f"Error processing {f.name}: {e}")
            
            if results:
                df = pd.DataFrame(results).sort_values("Rank Score", ascending=False)
                df_filtered = df[df["Rank Score"] >= min_match].head(top_k)
                
                st.markdown("---")
                
                # Report Action Row
                r_col1, r_col2 = st.columns([3, 1])
                with r_col1:
                    st.header("📊 Ranking Results")
                with r_col2:
                    try:
                        pdf_bytes = generate_pdf_report(jd_txt, target_skills, results)
                        st.download_button(
                            label="📄 Download Full PDF Report",
                            data=bytes(pdf_bytes) if isinstance(pdf_bytes, bytearray) else (pdf_bytes.encode('latin-1') if isinstance(pdf_bytes, str) else pdf_bytes),
                            file_name=f"Screening_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Reporting Error: {e}")
                
                # Visual highlight
                if not df_filtered.empty:
                    top_cand = df_filtered.iloc[0]
                    h_col1, h_col2, h_col3 = st.columns(3)
                    h_col1.metric("Top Fit", top_cand["Candidate"], f"{top_cand['Rank Score']}%")
                    h_col2.metric("Dominant Role", top_cand["Predicted Role"], f"{top_cand['ML Conf %']}% Conf")
                    h_col3.metric("Talent Pool", f"{len(df)} Resumes", "Processed")

                    st.dataframe(df_filtered.drop(columns=["Skills", "Gaps"]), use_container_width=True)
                    
                    # Skill Gap Detail
                    st.subheader("💡 Deep Dive: Skill Gaps")
                    for _, row in df_filtered.iterrows():
                        with st.expander(f"Analysis: {row['Candidate']} (Score: {row['Rank Score']})"):
                            c_a, c_b = st.columns(2)
                            with c_a:
                                st.markdown("**Skills Found:**")
                                if row['Skills']:
                                    for s in row['Skills']: st.markdown(f'<span class="skill-chip">{s}</span>', unsafe_allow_html=True)
                                else: st.write("None detected.")
                            with c_b:
                                st.markdown("**Missing Skills:**")
                                if row['Gaps']:
                                    for g in row['Gaps']: st.markdown(f'<span class="gap-chip">{g}</span>', unsafe_allow_html=True)
                                else: st.success("No critical skill gap!")
                    
                    # Chart
                    st.subheader("📈 Competitive Landscape")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    fig.patch.set_facecolor('#0f172a')
                    ax.set_facecolor('#1e293b')
                    sns.barplot(x="Rank Score", y="Candidate", data=df_filtered, palette="flare", ax=ax)
                    ax.tick_params(colors='white')
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    st.pyplot(fig)
                else:
                    st.warning(f"No candidates met the minimum match threshold of {min_match}%.")
                    st.dataframe(df.drop(columns=["Skills", "Gaps"]), use_container_width=True)
            else:
                st.warning("Could not extract enough data for ranking.")

st.markdown("---")
st.caption("ProScreen AI v6.0 • Advanced Reporting & LinkedIn Support • Task 3 Complete")
