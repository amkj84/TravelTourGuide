import os
import gdown
from groq import Groq
import gradio as gr
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from fpdf import FPDF

# Function to download PDFs from Google Drive links
def download_pdfs(links):
    pdf_paths = []
    for url in links:
        try:
            file_id = url.split('/d/')[1].split('/')[0]
            download_url = f'https://drive.google.com/uc?id={file_id}'
            output_path = f'./tourism_guide_{file_id}.pdf'
            gdown.download(download_url, output_path, quiet=False)
            pdf_paths.append(output_path)
        except Exception as e:
            print(f"Failed to download file from {url}: {e}")
    return pdf_paths

# List of Google Drive links
links = [
    'https://drive.google.com/file/d/1jwa2DhOVFysiCqZabUvJhSyPa9wHhuUB',
    'https://drive.google.com/file/d/1_D0st59XFBotuTehga0IvD6ah5jpYPRY'
]

# Download the PDFs
pdf_paths = download_pdfs(links)

# Extract text from all PDFs
pdf_text = ""
for path in pdf_paths:
    try:
        reader = PdfReader(path)
        pdf_text += "\n".join([page.extract_text() for page in reader.pages])
    except Exception as e:
        print(f"Failed to read PDF {path}: {e}")

# Chunking and embedding setup
chunks = [pdf_text[i:i+500] for i in range(0, len(pdf_text), 500)]
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)
normalized_embeddings = normalize(embeddings)

# Create FAISS index
dim = normalized_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(normalized_embeddings))

# Set up Groq API
client = Groq(api_key=os.getenv("APIkey"))

def generate_response(query):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": query}],
            model="llama-3.3-70b-versatile"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

def save_itinerary_to_pdf(days, region, location):
    # Generate query and response
    query = (
    f"Create a comprehensive {days}-days travel itinerary starting from {location} focusing on {region}. "
    "Structure your response with these elements:\n\n"
    
    "1. **Title**: [Region] Travel Itinerary\n"
    "2. **Introduction**: Brief overview of regional highlights and travel philosophy\n"
    "3. **Daily Plans** (Morning /Afternoon /Evening sections for each day):\n"
    "   - Must-see attractions with brief historical/cultural context\n"
    "   - Cultural immersion activities (local festivals, workshops, etc.)\n"
    "   - Adventure/outdoor experiences (hiking, water sports, etc.)\n"
    "   - Dining recommendations (3 options per meal: local specialties, popular cafes, fine dining)\n"
    "   - Travel time estimates between locations\n\n"
    
    "4. **Accommodation Guide**:\n"
    "   - Budget (<5000 PKR/night): Guesthouses, hostels\n"
    "   - Mid-Range (5000-15000 PKR/night): 3-4 star hotels\n"
    "   - Luxury (15000+ PKR/night): 5-star/resorts\n"
    "   - Include unique stays (homestays, heritage properties)\n\n"
    
    "5. **Transportation Matrix**:\n"
    "   - Inter-city options (bus\train\rental costs)\n"
    "   - Local transport (ride-hailing, rickshaws, costs)\n"
    "   - Parking/toll information if self-driving\n\n"
    
    "6. **Budget Breakdown**:\n"
    "   - Daily estimates (accommodation\meals\activities)\n"
    "   - Total projected costs in PKR\n"
    "   - Money-saving tips\n\n"
    
    "7. **Practical Information**:\n"
    "   - Packing checklist for climate/activities\n"
    "   - Cultural etiquette (dress codes, photography rules)\n"
    "   - Safety considerations\n"
    "   - Emergency contacts\n\n"
    
    "8. **Local Insights**:\n"
    "   - Best times to visit attractions\n"
    "   - Hidden gems off tourist trails\n"
    "   - Seasonal variations in experiences\n\n"
    
    "Format requirements:\n"
    "- Use clear section headings\n"
    "- Bullet points for lists\n"
    "- Tables for comparisons\n"
    "- Bold key numbers\names\n"
    "- Avoid markdown formatting\n"
    "- Maintain conversational but professional tone\n"
    "- Ensure logical geographical progression each day"
)
    response = generate_response(query)
    
    # Create PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add content
    pdf.cell(200, 10, txt=f"{region} Travel Itinerary", ln=True, align="C")
    pdf.ln(10)
    pdf.multi_cell(0, 10, response)
    
    # Save file
    file_name = f"{region.replace(' ', '_')}_Itinerary.pdf"
    pdf.output(file_name)
    return file_name, response

# Gradio Interface
def tourism_suggestions(days, region, location):
    itinerary_file, suggestions = save_itinerary_to_pdf(days, region, location)
    return suggestions, itinerary_file

interface = gr.Interface(
    fn=tourism_suggestions,
    inputs=[
        gr.Slider(minimum=1, maximum=30, step=1, label="Number of Days"),
        gr.Dropdown(choices=["Punjab", "KPK", "GB", "SINDH", "AJK", "Balochistan"], label="Region"),
        gr.Textbox(label="Current Location")
    ],
    outputs=[
        gr.Textbox(label="Tourism Suggestions"),
        gr.File(label="Download Itinerary PDF")
    ]
)

interface.launch()
