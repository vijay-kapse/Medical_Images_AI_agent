import os
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
import streamlit as st

# Load API Key securely from environment variables
#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = "AIzaSyD339SNnqvBhRpxCWL9Ln5WVcOQzQptufY"
if not GOOGLE_API_KEY:
    raise ValueError("⚠️ Missing Google API Key. Set it as an environment variable.")


from google import genai

# Initialize the client correctly
try:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    medical_agent = Agent(
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGoTools()],
        markdown=True
    )
except Exception as e:
    st.error(f"⚠️ Failed to initialize Gemini API: {e}")
    st.stop()




# # Initialize the AI agent
# medical_agent = Agent(
#     model=Gemini(id="gemini-2.0-flash-exp"),
#     tools=[DuckDuckGoTools()],
#     markdown=True,
# )

# Prompt for medical image analysis
ANALYSIS_PROMPT = """
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

# Function to analyze the medical image
def analyze_medical_image(image_path):
    """Processes and analyzes a medical image using AI."""
    try:
        # Open and resize the image for consistent processing
        image = PILImage.open(image_path)
        width, height = image.size
        aspect_ratio = width / height
        new_width = 500
        new_height = int(new_width / aspect_ratio)
        resized_image = image.resize((new_width, new_height))

        # Save resized image temporarily for analysis
        temp_path = "temp_resized_image.png"
        resized_image.save(temp_path)

        # Create AgnoImage object for AI analysis
        agno_image = AgnoImage(filepath=temp_path)

        # Run the AI agent with the prompt and image
        response = medical_agent.run(ANALYSIS_PROMPT, images=[agno_image])

        # Clean up temporary file after processing
        os.remove(temp_path)

        return response.content

    except Exception as e:
        return f"⚠️ Analysis error: {e}"

# Streamlit App UI Design
st.set_page_config(page_title="Medical Image Analysis", layout="wide")
st.title("🩺 Medical Image Analysis Tool 🔬")
st.markdown(
    """
    Welcome to the **Medical Image Analysis Tool**! 📸  
    Upload a medical image (X-ray, MRI, CT, Ultrasound, etc.), and let our AI-powered system analyze it for you.  
    You'll receive detailed findings, diagnostic assessments, and research insights in seconds!  
    """
)

# Sidebar for uploading images
st.sidebar.header("Upload Your Medical Image:")
uploaded_file = st.sidebar.file_uploader(
    "Choose a medical image file", type=["jpg", "jpeg", "png", "bmp", "gif"]
)

# Main section for displaying results or errors
if uploaded_file:
    try:
        # Display uploaded image preview in Streamlit
        st.image(uploaded_file, caption="Uploaded Image Preview", use_column_width=True)

        # Button to trigger analysis
        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing the image... Please wait ⏳"):
                # Save uploaded file temporarily for processing
                temp_image_path = f"temp_uploaded_image.{uploaded_file.type.split('/')[1]}"
                with open(temp_image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Run the analysis function on the uploaded image
                report = analyze_medical_image(temp_image_path)

                # Display the analysis report in markdown format
                st.subheader("📋 Analysis Report")
                st.markdown(report, unsafe_allow_html=True)

                # Add a download button for the report (optional)
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name="medical_analysis_report.md",
                    mime="text/markdown",
                )

                # Clean up temporary file after use
                os.remove(temp_image_path)

    except Exception as e:
        st.error(f"⚠️ An error occurred while processing your image: {e}")
else:
    st.info("ℹ️ Please upload a medical image to begin analysis.")

# Footer Section (Optional)
st.markdown("---")
st.markdown(
    """
    **Disclaimer:** This tool is intended for educational purposes only.  
    Consult a licensed healthcare professional for any medical concerns or diagnoses.
"""
)
