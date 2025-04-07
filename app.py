import os
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Environment & Agent Setup
# ──────────────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("⚠️ Please set your Google API Key in GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

medical_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGoTools()],
    markdown=True
)

query = """
You are a highly skilled medical imaging expert... [shortened for clarity]
"""

# ──────────────────────────────────────────────────────────────────────────────
# AI Image Analysis Function
# ──────────────────────────────────────────────────────────────────────────────
def analyze_medical_image(image_path):
    try:
        image = PILImage.open(image_path)
        aspect_ratio = image.width / image.height
        resized_image = image.resize((500, int(500 / aspect_ratio)))
        resized_image.save("temp_resized_image.png")

        agno_image = AgnoImage(filepath="temp_resized_image.png")
        response = medical_agent.run(query, images=[agno_image])
        return response.content
    except Exception as e:
        return f"⚠️ Error during analysis: {e}"
    finally:
        if os.path.exists("temp_resized_image.png"):
            os.remove("temp_resized_image.png")

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Medical Imaging AI", layout="wide", page_icon="🧠")

# Inject custom styling
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .report-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin-top: 20px;
        }
        .sidebar .sidebar-content {
            background-color: #2c3e50;
            color: white;
        }
        .css-1cpxqw2, .st-bf, .st-cg, .st-ci, .st-ch {
            color: white !important;
        }
        h1, h2, h3 {
            color: #1e3d59;
        }
        .stButton > button {
            background-color: #007bff;
            color: white;
            border-radius: 6px;
            padding: 10px 20px;
            font-weight: bold;
            margin-top: 10px;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
    </style>
""", unsafe_allow_html=True)

# Title section
st.title("🩺 AI-Powered Medical Imaging Analysis")
st.markdown("""
Upload any medical image — such as an **X-ray**, **CT**, **MRI**, or **Ultrasound** — and our AI system will provide an **expert-level diagnostic analysis** in seconds.

🔒 _Your data stays private. Nothing is stored._
""")

# Layout: Sidebar for image input, main column for output
left_col, right_col = st.columns([1, 2])

# Sidebar upload
with left_col:
    st.subheader("📤 Upload Medical Image")
    uploaded_file = st.file_uploader("Supported formats: JPG, PNG, BMP, GIF", type=["jpg", "jpeg", "png", "bmp", "gif"])

    if uploaded_file and st.button("🔍 Analyze Image"):
        # Save uploaded file temporarily
        file_ext = uploaded_file.type.split('/')[-1]
        temp_path = f"temp_image.{file_ext}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Main panel: image and result
        with right_col:
            st.image(uploaded_file, caption="🖼 Uploaded Image", use_column_width=True)
            with st.spinner("Analyzing the image..."):
                analysis = analyze_medical_image(temp_path)
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            with st.expander("📋 View Full Diagnostic Report", expanded=True):
                st.markdown(analysis, unsafe_allow_html=True)

    elif not uploaded_file:
        st.info("Please upload a medical image to begin.")
