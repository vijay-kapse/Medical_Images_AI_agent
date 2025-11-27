import os
import re
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
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
# Switched to gemini-1.5-flash for stability
# Removed DuckDuckGoTools to prevent 429 Rate Limit errors
# -------------------------------------------------------------------
# Initialize Medical Agent
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Initialize Medical Agent (with defensive error handling)
# -------------------------------------------------------------------
def make_agent():
    candidate_model_ids = [
        "gemini-pro-vision",
        "gemini-1.0-pro-vision",
        "gemini-1.5-pro", 
        # "gemini-1.5-flash",  # will work AFTER billing enabled
    ]
    last_exc = None
    for mid in candidate_model_ids:
        try:
            agent = Agent(model=Gemini(id=mid), markdown=True)
            st.sidebar.success(f"Using model: {mid}")
            return agent
        except Exception as e:
            last_exc = e
            st.sidebar.warning(f"Model {mid} init failed: {e}")
    raise RuntimeError(f"Failed to initialize any model.\nLast error:\n{last_exc}")


try:
    medical_agent = make_agent()
except Exception as init_err:
    # Show error in UI and keep app running
    st.error("Could not initialize medical agent. See sidebar for details.")
    st.sidebar.error(str(init_err))
    # Save for analyze_medical_image to return an informative message
    medical_agent = None


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

### 5. Medical Context & Next Steps
- Suggest standard clinical next steps (e.g., referral, biopsy, follow-up imaging).
- Provide general treatment protocols for the identified condition based on standard medical guidelines.

Ensure a structured and medically accurate response using clear markdown formatting.
"""

# -------------------------------------------------------------------
# Analyze Image Function
# -------------------------------------------------------------------
def analyze_medical_image(image_path):
    temp_path = "temp_resized_image.png"
    try:
        image = PILImage.open(image_path)
        
        # Resize logic to save tokens and speed up upload
        aspect_ratio = image.width / image.height
        new_width = 500
        new_height = int(new_width / aspect_ratio)
        resized_image = image.resize((new_width, new_height))
        
        resized_image.save(temp_path)
        
        # Pass the image to the agent
        agno_image = AgnoImage(filepath=temp_path)
        response = medical_agent.run(query, images=[agno_image])
        return response.content
        
    except Exception as e:
        return f"⚠️ Analysis error: {e}"
        
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# -------------------------------------------------------------------
# Summary Extractor
# -------------------------------------------------------------------
def extract_summary(report):
    try:
        # Attempt to grab the text after the Diagnostic Assessment header
        if "### 3. Diagnostic Assessment" in report:
            diagnostic_section = report.split("### 3. Diagnostic Assessment")[1]
        else:
            diagnostic_section = report

        # Look for "Primary Diagnosis:" pattern
        match = re.search(r"Primary Diagnosis\s*:\s*(.*)", diagnostic_section, re.IGNORECASE)
        if match:
            # Return just the first line of the diagnosis
            return match.group(1).split('\n')[0].strip()
            
    except Exception:
        pass

    return "See detailed report below."

# -------------------------------------------------------------------
# UI Setup
# -------------------------------------------------------------------
st.set_page_config(page_title="Medical Image Analysis", layout="centered")

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
        h1 { color: #045d75; }
        .stMarkdown { font-size: 1rem; line-height: 1.6; }
        div[data-testid="stImage"] img { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Title & Sidebar
# -------------------------------------------------------------------
st.title("🩺 Medical Image Analysis Tool 🔬")
st.markdown("Upload a medical image for a structured AI diagnostic report.")

st.sidebar.header("📤 Upload Medical Image")
uploaded_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png", "bmp", "gif"])

# -------------------------------------------------------------------
# Logic
# -------------------------------------------------------------------
def handle_image(image_path, caption="Uploaded Image"):
    with st.spinner("🔍 Running analysis..."):
        report = analyze_medical_image(image_path)
        summary = extract_summary(report)

        col1, col2 = st.columns([1, 2])
        with col1:
            # Fixed warning by using width="stretch"
            st.image(image_path, caption=f"🖼️ {caption}", width="stretch")

        with col2:
            st.subheader("📋 Report")
            if "⚠️" not in report:
                st.info(f"**Diagnostic Summary:** {summary}")
            st.markdown(report, unsafe_allow_html=True)

# Handle Upload
if uploaded_file is not None:
    file_extension = uploaded_file.type.split('/')[-1]
    image_path = f"temp_image.{file_extension}"
    
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    handle_image(image_path)
    
    if os.path.exists(image_path):
        os.remove(image_path)
else:
    st.info("Please upload an image to begin.")

# Test Images (Optional)
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Test Data")
if st.sidebar.button("Load Test Image 1"):
    if os.path.exists("test_images/test1.png"):
        handle_image("test_images/test1.png", caption="Test Case 1")
    else:
        st.sidebar.error("test_images/test1.png not found.")

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown("""
<hr style="margin-top: 2rem;">
<div style='text-align: center; color: #555;'>
    Made with ❤️ by <a href="https://www.linkedin.com/in/vijay-kapse/" target="_blank">Vijay Suryakant Kapse</a>
</div>
""", unsafe_allow_html=True)
