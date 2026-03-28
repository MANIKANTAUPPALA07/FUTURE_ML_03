from fpdf import FPDF
import datetime

class ScreeningReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.set_text_color(30, 58, 138) # Dark blue
        self.cell(0, 10, 'ProScreen AI - Advanced Screening Report', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.set_text_color(100, 116, 139) # Gray
        self.cell(0, 5, f'Generated on: {datetime.datetime.now().strftime("%Y-%m-d %H:%M")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_section_header(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_text_color(59, 130, 246) # Bright blue
        self.cell(0, 10, title.upper(), 0, 1, 'L')
        self.set_draw_color(226, 232, 240)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

def generate_pdf_report(jd_text, target_skills, results):
    pdf = ScreeningReport()
    pdf.add_page()
    
    # 1. Job Description Summary
    pdf.add_section_header("Job Requirement Overview")
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(40, 7, "Extracted Target Skills:", 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, ", ".join(target_skills) if target_skills else "General screening", 0, 'L')
    pdf.ln(5)

    # 2. Results Table
    pdf.add_section_header("Candidate Rankings")
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(241, 245, 249)
    pdf.cell(80, 10, 'Candidate Name', 1, 0, 'C', True)
    pdf.cell(50, 10, 'Predicted Role', 1, 0, 'C', True)
    pdf.cell(30, 10, 'Fit Score', 1, 0, 'C', True)
    pdf.cell(30, 10, 'Confidence', 1, 1, 'C', True)
    
    pdf.set_font('Arial', '', 10)
    for res in sorted(results, key=lambda x: x['Rank Score'], reverse=True):
        pdf.cell(80, 10, str(res['Candidate'])[:40], 1, 0, 'L')
        pdf.cell(50, 10, str(res['Predicted Role']), 1, 0, 'C')
        pdf.cell(30, 10, f"{res['Rank Score']}%", 1, 0, 'C')
        pdf.cell(30, 10, f"{res['ML Conf %']}%", 1, 1, 'C')
    
    pdf.add_page()
    # 3. Detailed Gaps
    pdf.add_section_header("In-Depth Skill Analysis")
    for res in results:
        pdf.set_font('Arial', 'B', 11)
        pdf.set_text_color(30, 58, 138)
        pdf.cell(0, 10, f"Candidate: {res['Candidate']}", 0, 1)
        
        pdf.set_font('Arial', 'B', 9)
        pdf.set_text_color(22, 101, 52) # Dark green
        pdf.cell(35, 6, "Skills Found:", 0, 0)
        pdf.set_font('Arial', '', 9)
        pdf.multi_cell(155, 6, ", ".join(res['Skills']) if res['Skills'] else "None detected", 0, 'L')
        
        pdf.set_font('Arial', 'B', 9)
        pdf.set_text_color(153, 27, 27) # Dark red
        pdf.cell(35, 6, "Skill Gaps:", 0, 0)
        pdf.set_font('Arial', '', 9)
        pdf.multi_cell(155, 6, ", ".join(res['Gaps']) if res['Gaps'] else "No critical gaps identified", 0, 'L')
        
        pdf.ln(5)
        pdf.set_draw_color(241, 245, 249)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(5)
        
        if pdf.get_y() > 250: pdf.add_page()

    return pdf.output()
