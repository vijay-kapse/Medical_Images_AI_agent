import os
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
import streamlit as st

# -------------------------------------------------------------------
# Environment Setup
# -------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("⚠️ Please set your Google API Key in GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# -------------------------------------------------------------------
# Initialize Medical Agent
# -------------------------------------------------------------------
medical_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGoTools()],
    markdown=True
)

# -------------------------------------------------------------------
# Define the Analysis Query
# -------------------------------------------------------------------
query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the medical image and structure your response as follows:

### 1. Image Type & Region
- Identify imaging modality (X-ray/MRI/CT/Ultrasound/etc.).
- Specify anatomical region and positioning.
- Evaluate image quality and technical adequacy.

### 2. Key Findings
- Highlight primary observations systematically.
- Identify potential abnormalities with detailed descriptions.
- Include measurements and densities where relevant.

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level.
- List differential diagnoses ranked by likelihood.
- Support each diagnosis with observed evidence.
- Highlight critical/urgent findings.

### 4. Patient-Friendly Explanation
- Simplify findings in clear, non-technical language.
- Avoid medical jargon or provide easy definitions.
- Include relatable visual analogies.

### 5. Research Context
- Use DuckDuckGo search to find recent medical literature.
- Search for standard treatment protocols.
- Provide 2-3 key references supporting the analysis.

Ensure a structured and medically accurate response using clear markdown formatting.
"""

# -------------------------------------------------------------------
# Function: Analyze Medical Image
# -------------------------------------------------------------------
def analyze_medical_image(image_path):
    try:
        image = PILImage.open(image_path)
        aspect_ratio = image.width / image.height
        resized = image.resize((500, int(500 / aspect_ratio)))

        temp_resized_path = "temp_resized_image.png"
        resized.save(temp_resized_path)

        agno_image = AgnoImage(filepath=temp_resized_path)
        response = medical_agent.run(query, images=[agno_image])
        return response.content
    except Exception as e:
        return f"⚠️ Analysis error: {e}"
    finally:
        if os.path.exists("temp_resized_image.png"):
            os.remove("temp_resized_image.png")

# -------------------------------------------------------------------
# Streamlit UI Setup
# -------------------------------------------------------------------
st.set_page_config(page_title="Medical Image Analysis", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f9fbfc;
        }
        .main {
            padding: 2rem;
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        }
        .stSidebar {
            background-color: #ecf0f3;
        }
        h1 {
            color: #045d75;
        }
        .stMarkdown {
            font-size: 1rem;
            line-height: 1.6;
        }
    </style>
""", unsafe_allow_html=True)

# Page Header
st.title("🩺 AI-Powered Medical Image Analyzer")
st.markdown("""
Upload your **radiological image** below (e.g., X-ray, CT, MRI, Ultrasound) and receive a detailed, AI-generated diagnostic report.
""")

# Sidebar
st.sidebar.header("📤 Upload Medical Image")
uploaded_file = st.sidebar.file_uploader("Supported formats: JPG, PNG, BMP, GIF", type=["jpg", "jpeg", "png", "bmp", "gif"])

# Image Handling & Analysis
if uploaded_file is not None:
    file_extension = uploaded_file.type.split("/")[-1]
    temp_image_path = f"temp_uploaded_image.{file_extension}"

    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Layout: Show image on left, result on right
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(uploaded_file, caption="📸 Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("🧠 Analyzing image with medical AI..."):
            report = analyze_medical_image(temp_image_path)
            st.markdown("### 📋 Diagnostic Report")
            st.markdown(report, unsafe_allow_html=True)

    # Clean up uploaded temp file
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)
else:
    st.info("Upload a medical image from the sidebar to begin the analysis.")
