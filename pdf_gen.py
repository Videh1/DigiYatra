import markdown
from fpdf import FPDF

class ItineraryPDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 15)
        self.cell(0, 10, "DigiYatra Master Itinerary", border=0, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def clean_text_for_pdf(text: str) -> str:
    """
    Strips complex emojis and unsupported characters that crash FPDF.
    Allows basic punctuation and alphanumeric characters.
    """
    # Encoding to ascii and ignoring errors safely strips emojis
    return text.encode('ascii', 'ignore').decode('ascii')

def generate_itinerary_pdf(username: str, master_itinerary: str) -> bytes:
    """
    Generates a PDF byte stream containing the beautifully formatted Master Itinerary.
    """
    pdf = ItineraryPDF()
    pdf.add_page()
    
    if master_itinerary:
        safe_plan = clean_text_for_pdf(master_itinerary)
        html_plan = markdown.markdown(safe_plan)
        pdf.write_html(html_plan)
    else:
        pdf.set_font("helvetica", size=11)
        pdf.multi_cell(0, 8, "No itinerary generated yet.")

    # Return raw PDF bytes
    return bytes(pdf.output())
