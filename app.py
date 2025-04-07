import os
import time
import base64
from datetime import datetime
from pathlib import Path
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
from dotenv import load_dotenv
from io import BytesIO
import json

# Attempt to import Agno modules with error handling
try:
    from agno.agent import Agent
    from agno.models.google import Gemini
    from agno.tools.duckduckgo import DuckDuckGoTools
    from agno.media import Image as AgnoImage
except ImportError:
    st.error("⚠️ Agno packages not found. Please install with: pip install agno-ai")
    st.stop()

# Configuration and Utilities
class Config:
    """Application configuration and utilities"""
    
    # Constants
    TEMP_DIR = Path("temp")
    CACHE_DIR = Path("cache")
    SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "bmp", "tiff", "dicom"]
    
    @staticmethod
    def initialize():
        """Initialize application environment"""
        # Load environment variables
        load_dotenv()
        
        # Create necessary directories
        Config.TEMP_DIR.mkdir(exist_ok=True)
        Config.CACHE_DIR.mkdir(exist_ok=True)
        
        # Check for API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.sidebar.warning("⚠️ Google API key not found in environment")
            api_key = st.sidebar.text_input("Enter Google API Key:", type="password")
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
                return True
            return False
        return True
    
    @staticmethod
    def clean_temp_files():
        """Remove temporary files"""
        for file in Config.TEMP_DIR.glob("*"):
            try:
                file.unlink()
            except Exception as e:
                print(f"Error removing {file}: {e}")

# Medical Image Analyzer
class MedicalImageAnalyzer:
    """Handles medical image analysis using the Agno framework"""
    
    def __init__(self):
        """Initialize the analyzer with the appropriate model and tools"""
        try:
            self.agent = Agent(
                model=Gemini(id="gemini-2.0-flash-exp"),
                tools=[DuckDuckGoTools()],
                markdown=True
            )
            self.ready = True
        except Exception as e:
            st.error(f"Failed to initialize analyzer: {e}")
            self.ready = False
    
    def get_analysis_prompt(self):
        """Return the medical analysis prompt"""
        return """
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
    
    def preprocess_image(self, image_path):
        """Enhance and prepare image for analysis"""
        try:
            # Open and enhance image
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            
            # Basic enhancements
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)  # Slightly increase contrast
            
            # Preserve aspect ratio while resizing
            width, height = img.size
            aspect_ratio = width / height
            new_width = 800  # Higher resolution than original
            new_height = int(new_width / aspect_ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save processed image
            processed_path = Config.TEMP_DIR / "processed_image.png"
            img.save(processed_path)
            
            return processed_path
        except Exception as e:
            st.error(f"Image preprocessing failed: {e}")
            return None
    
    def analyze(self, image_path):
        """Run analysis on the provided image"""
        if not self.ready:
            return "Analyzer not properly initialized."
        
        try:
            # Preprocess the image
            processed_path = self.preprocess_image(image_path)
            if not processed_path:
                return "Image preprocessing failed."
            
            # Create AgnoImage object
            agno_image = AgnoImage(filepath=str(processed_path))
            
            # Run analysis
            with st.spinner("🔍 Analyzing image with AI... This might take a minute."):
                start_time = time.time()
                response = self.agent.run(self.get_analysis_prompt(), images=[agno_image])
                analysis_time = time.time() - start_time
                
                # Cache result
                result = {
                    "content": response.content,
                    "timestamp": datetime.now().isoformat(),
                    "analysis_time": analysis_time
                }
                
                return result
                
        except Exception as e:
            return f"⚠️ Analysis failed: {str(e)}"
        finally:
            # Clean up
            Config.clean_temp_files()

# UI Components
class UI:
    """UI components and handlers"""
    
    @staticmethod
    def setup_page():
        """Configure page settings and layout"""
        st.set_page_config(
            page_title="Advanced Medical Image Analysis",
            page_icon="🩺",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply custom CSS
        st.markdown("""
        <style>
            .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
            .stTabs [data-baseweb="tab-list"] {gap: 8px;}
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre-wrap;
                background-color: #f0f2f6;
                border-radius: 4px 4px 0px 0px;
                gap: 1px;
                padding-top: 10px;
                padding-bottom: 10px;
            }
            .stTabs [aria-selected="true"] {
                background-color: #e0e5f0;
                border-bottom: 2px solid #4b6cb7;
            }
            div.stButton > button:first-child {
                background-color: #4b6cb7;
                color: white;
                font-weight: bold;
                padding: 0.5rem 1rem;
                border-radius: 0.5rem;
            }
            div.stButton > button:hover {
                background-color: #6983c7;
                color: white;
            }
            .report-container {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .success-message {
                background-color: #d4edda;
                color: #155724;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .info-box {
                background-color: #e6f3ff; 
                border-left: 3px solid #4b6cb7;
                padding: 10px 15px;
                margin: 10px 0px;
                border-radius: 3px;
            }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_header():
        """Render the application header"""
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
        with col2:
            st.title("Advanced Medical Image Analysis")
            st.markdown("""
            <div class="info-box">
                Upload medical images (X-ray, MRI, CT, etc.) for AI-powered analysis and diagnostics.
                This tool provides detailed findings, potential diagnoses, and relevant medical research.
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def render_sidebar():
        """Render the sidebar content"""
        st.sidebar.header("📤 Upload Medical Image")
        
        # File uploader
        uploaded_file = st.sidebar.file_uploader(
            "Choose a medical image file",
            type=Config.SUPPORTED_FORMATS,
            help="Upload X-ray, MRI, CT scan, or other medical imaging files"
        )
        
        # Sample images
        st.sidebar.markdown("---")
        st.sidebar.subheader("Or try a sample image:")
        sample_options = {
            "None": None,
            "Chest X-ray": "samples/chest_xray.jpg",
            "Brain MRI": "samples/brain_mri.jpg",
            "Knee MRI": "samples/knee_mri.jpg"
        }
        sample_choice = st.sidebar.selectbox("Select sample:", list(sample_options.keys()))
        sample_file = sample_options[sample_choice]
        
        # Settings and options
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ Settings")
        show_confidence = st.sidebar.checkbox("Show diagnostic confidence", value=True)
        include_references = st.sidebar.checkbox("Include research references", value=True)
        
        # Information section
        st.sidebar.markdown("---")
        st.sidebar.info("""
        **Disclaimer:** This tool is for educational purposes only. 
        It's not a substitute for professional medical advice or diagnosis.
        """)
        
        return uploaded_file, sample_file, {"show_confidence": show_confidence, "include_references": include_references}
    
    @staticmethod
    def display_image_panel(image_path):
        """Display the image with options"""
        st.subheader("📷 Uploaded Image")
        
        try:
            img = Image.open(image_path)
            st.image(img, use_column_width=True)
            
            # Image information
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Dimensions:** {img.width} × {img.height} pixels")
                st.markdown(f"**Format:** {img.format}")
            with col2:
                file_size = os.path.getsize(image_path) / 1024
                st.markdown(f"**File size:** {file_size:.1f} KB")
                st.markdown(f"**Mode:** {img.mode}")
        except Exception as e:
            st.error(f"Cannot display image: {e}")
    
    @staticmethod
    def display_analysis_report(result):
        """Display the analysis report with formatting"""
        if not result or "content" not in result:
            st.error("Analysis result is missing or incomplete")
            return
        
        st.header("📋 Analysis Report")
        
        # Show analysis metadata
        if "analysis_time" in result:
            st.markdown(f"*Analysis completed in {result['analysis_time']:.1f} seconds*")
        
        # Display the formatted report
        st.markdown("<div class='report-container'>", unsafe_allow_html=True)
        st.markdown(result["content"])
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Create download button for the report
        report_md = result["content"]
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Medical Image Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .report-date {{ color: #7f8c8d; font-size: 0.9em; }}
                .disclaimer {{ background-color: #f8f9fa; padding: 10px; border-left: 3px solid #d35400; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <h1>Medical Image Analysis Report</h1>
            <p class="report-date">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
            {report_md}
            <div class="disclaimer">
                <p><strong>Disclaimer:</strong> This analysis is generated by AI and is for informational purposes only. 
                It is not a substitute for professional medical advice, diagnosis, or treatment.</p>
            </div>
        </body>
        </html>
        """
        
        b64_html = base64.b64encode(report_html.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64_html}" download="medical_analysis_report.html">Download Full Report</a>'
        st.markdown(href, unsafe_allow_html=True)

# Main Application
def main():
    """Main application entry point"""
    # Initialize
    UI.setup_page()
    
    # Check configuration and environment
    if not Config.initialize():
        st.warning("Please set up your API key to continue.")
        st.stop()
    
    # Create analyzer
    analyzer = MedicalImageAnalyzer()
    if not analyzer.ready:
        st.error("Failed to initialize the analysis system.")
        st.stop()
    
    # Render header
    UI.render_header()
    
    # Render sidebar and get inputs
    uploaded_file, sample_file, options = UI.render_sidebar()
    
    # Main content area
    tabs = st.tabs(["Image Analysis", "History", "About"])
    
    with tabs[0]:  # Image Analysis tab
        col1, col2 = st.columns([3, 5])
        
        with col1:
            image_path = None
            
            # Process uploaded file or sample
            if uploaded_file:
                # Save uploaded file
                image_path = Config.TEMP_DIR / uploaded_file.name
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success("Image uploaded successfully!")
            elif sample_file:
                image_path = sample_file
            
            # Display image if available
            if image_path:
                UI.display_image_panel(image_path)
                
                # Analysis button
                if st.button("🔍 Analyze Image", use_container_width=True):
                    if not os.path.exists(image_path):
                        st.error("The image file is not accessible.")
                    else:
                        # Run analysis
                        result = analyzer.analyze(image_path)
                        
                        # Store in session state
                        if "analysis_history" not in st.session_state:
                            st.session_state.analysis_history = []
                        
                        if isinstance(result, dict) and "content" in result:
                            # Add to history
                            history_item = {
                                "timestamp": datetime.now().isoformat(),
                                "filename": os.path.basename(str(image_path)),
                                "result": result
                            }
                            st.session_state.analysis_history.insert(0, history_item)
                            
                            # Display results in the right column
                            with col2:
                                UI.display_analysis_report(result)
                        else:
                            with col2:
                                st.error(f"Analysis failed: {result}")
                
                # If results already exist in session
                elif "analysis_history" in st.session_state and st.session_state.analysis_history:
                    # Check if current image matches the latest analyzed one
                    latest = st.session_state.analysis_history[0]
                    if latest["filename"] == os.path.basename(str(image_path)):
                        with col2:
                            UI.display_analysis_report(latest["result"])
            else:
                st.info("Please upload a medical image or select a sample to begin.")
        
    with tabs[1]:  # History tab
        st.header("Analysis History")
        
        if "analysis_history" not in st.session_state or not st.session_state.analysis_history:
            st.info("No analysis history available. Analyze an image to see results here.")
        else:
            # Display history entries
            for i, entry in enumerate(st.session_state.analysis_history):
                with st.expander(f"Analysis {i+1}: {entry['filename']} - {entry['timestamp'][:16].replace('T', ' ')}"):
                    UI.display_analysis_report(entry["result"])
                    
                    # Option to clear this entry
                    if st.button(f"Clear Entry #{i+1}", key=f"clear_{i}"):
                        st.session_state.analysis_history.pop(i)
                        st.rerun()
            
            # Clear all history button
            if st.button("Clear All History"):
                st.session_state.analysis_history = []
                st.rerun()
    
    with tabs[2]:  # About tab
        st.header("About This Tool")
        st.markdown("""
        ## Advanced Medical Image Analysis System
        
        This application uses state-of-the-art AI to analyze medical images and provide detailed assessments.
        
        ### Features:
        - Multiple medical imaging modality support (X-ray, MRI, CT, Ultrasound)
        - Detailed structured analysis with findings and diagnostic possibilities
        - Patient-friendly explanations of medical terminology
        - Integration with medical research literature
        - Report generation and download
        
        ### How It Works:
        1. Upload a medical image or select a sample
        2. The system preprocesses and enhances the image for optimal analysis
        3. The AI model (Google Gemini 2.0) analyzes the image content
        4. Results are presented in a structured format with diagnostic insights
        5. Additional research context is provided through online search integration
        
        ### Technology Stack:
        - Streamlit for the user interface
        - Google Gemini AI for image analysis
        - Agno framework for AI orchestration
        - PIL/Pillow for image processing
        
        ### Limitations:
        This tool is for educational and informational purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers.
        """)

# Run the application
if __name__ == "__main__":
    main()
