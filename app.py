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
# Medical Analysis Prompt
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
        # Resize image
        image = PILImage.open(image_path)
        width, height = image.size
        aspect_ratio = width / height
        new_width = 500
        new_height = int(new_width / aspect_ratio)
        resized_image = image.resize((new_width, new_height))

        # Save temp file
        temp_path = "temp_resized_image.png"
        resized_image.save(temp_path)

        # Analyze with agent
        agno_image = AgnoImage(filepath=temp_path)
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
st.set_page_config(page_title="Medical Image Analysis", layout="centered")

# -------------------------------------------------------------------
# Custom CSS for Better UI/UX
# -------------------------------------------------------------------
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

        section[data-testid="stSidebar"] {
            background-color: #ecf0f3;
            color: #1c1c1c;
        }

        section[data-testid="stSidebar"] .stTextInput,
        section[data-testid="stSidebar"] .stFileUploader,
        section[data-testid="stSidebar"] .stMarkdown {
            color: #1c1c1c !important;
        }

        .stMarkdown {
            font-size: 1rem;
            line-height: 1.6;
        }

        h1 {
            color: #045d75;
        }

        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-thumb {
            background-color: #ccc;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)



st.markdown("""
    <style>
        /* Style sidebar buttons */
        div[data-testid="stSidebar"] button {
            color: white !important;
            background-color: #045d75 !important;
            border: none !important;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }

        div[data-testid="stSidebar"] button:hover {
            background-color: #033d52 !important;
            transition: 0.2s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# App Title and Introduction
# -------------------------------------------------------------------
st.title("🩺 Medical Image Analysis Tool 🔬")
st.markdown("""
Welcome to the **Medical Image Analysis** tool powered by AI.<br>
Upload a medical image (X-ray, MRI, CT, Ultrasound, etc.) and receive a structured, detailed diagnostic report with insights, potential findings, and references to medical research.
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Sidebar Upload and Instructions
# -------------------------------------------------------------------
st.sidebar.header("📤 Upload Medical Image")
st.sidebar.markdown("""
**Instructions:**
- Supported formats: JPG, JPEG, PNG, BMP, GIF
- Upload valid radiology images
- Analysis starts automatically after upload
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png", "bmp", "gif"])

# -------------------------------------------------------------------
# Process Upload and Display Results
# -------------------------------------------------------------------
if uploaded_file is not None:
    st.image(uploaded_file, caption="🖼️ Uploaded Image", use_column_width=True)
    with st.spinner("🔍 Running analysis..."):
        # Save uploaded image
        file_extension = uploaded_file.type.split('/')[-1]
        image_path = f"temp_image.{file_extension}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Analyze image
        report = analyze_medical_image(image_path)

        # Display report
        st.subheader("📋 Analysis Report")
        st.markdown(report, unsafe_allow_html=True)

        # Cleanup
        if os.path.exists(image_path):
            os.remove(image_path)
else:
    st.warning("⚠️ Please upload a medical image to begin analysis.")


# -------------------------------------------------------------------
# Preloaded Test Images for Quick Demo
# -------------------------------------------------------------------
st.sidebar.markdown("### 🧪 Try with Test Images")

col1, col2 = st.sidebar.columns(2)
if col1.button("Test Image 1"):
    test_image_path = "test_images/test1.png"
    st.image(test_image_path, caption="🖼️ Test Image 1", use_column_width=True)
    with st.spinner("🔍 Running analysis on Test Image 1..."):
        report = analyze_medical_image(test_image_path)
        st.subheader("📋 Analysis Report")
        st.markdown(report, unsafe_allow_html=True)

if col2.button("Test Image 2"):
    test_image_path = "test_images/test2.png"
    st.image(test_image_path, caption="🖼️ Test Image 2", use_column_width=True)
    with st.spinner("🔍 Running analysis on Test Image 2..."):
        report = analyze_medical_image(test_image_path)
        st.subheader("📋 Analysis Report")
        st.markdown(report, unsafe_allow_html=True)


