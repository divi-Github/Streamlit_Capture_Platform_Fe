import streamlit as st
import pandas as pd
import requests
import json
from streamlit_code_diff import st_code_diff
import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO


# Streamlit page configuration
st.set_page_config(page_title="JSON Data Extraction Model", layout="wide")

# API base URL for FastAPI backend
API_BASE_URL = "https://tuboid-nonparochially-christian.ngrok-free.dev/data"  
# API_BASE_URL = "http://localhost:8000/data"  
MODEL3_API_BASE_URL = "https://7b544424febc.ngrok-free.app" 

# --- API Functions ---
def fetch_documents(model_type="local"):
    """
    Fetch documents from the appropriate API based on model_type.
    - 'local': Uses local API for Model-1 and Model-2
    - 'model3': Uses ngrok API for Model-3
    """
    if model_type == "model3":
        url = f"{MODEL3_API_BASE_URL}/data/"
        headers = {"accept": "application/json"}
    else:
        url = f"{API_BASE_URL}/"
        headers = {}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json().get("data", [])
        if not data:
            st.warning(f"No documents found in the {model_type} database.")
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch {model_type} documents: {str(e)}")
        return []

def fetch_all_documents():
    """
    Fetch documents from both local and Model-3 APIs and combine them.
    Add a 'source' field to distinguish between them.
    """
    local_docs = fetch_documents("local")
    model3_docs = fetch_documents("model3")
    
    # Add source field
    for doc in local_docs:
        doc['source'] = 'Model 1 & Model 2'
    for doc in model3_docs:
        doc['source'] = 'Model 3'
    
    all_docs = local_docs + model3_docs
    return all_docs

# --- JSON Comparison Functions ---
def flatten_json(y, prefix=''):
    out = {}
    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], f'{name}{a}.')
        elif isinstance(x, list):
            for i, a in enumerate(x):
                flatten(a, f'{name}{i}.')
        else:
            out[name[:-1]] = x
    flatten(y, prefix)
    return out

def compare_json(json1, json2):
    flat_json1 = flatten_json(json1)
    flat_json2 = flatten_json(json2)
    total_keys = len(flat_json1)
    matched_keys = sum(1 for k in flat_json1 if k in flat_json2 and flat_json1[k] == flat_json2[k])
    mismatched_keys = total_keys - matched_keys
    accuracy = (matched_keys / total_keys) * 100 if total_keys else 0
    mismatch_data = []
    for key in flat_json1:
        if key not in flat_json2:
            mismatch_data.append((key, flat_json1[key], "❌ Missing"))
        elif flat_json1[key] != flat_json2[key]:
            mismatch_data.append((key, flat_json1[key], flat_json2[key]))
    return {
        "matched_keys": matched_keys,
        "total_keys": total_keys,
        "mismatched_keys": mismatched_keys,
        "accuracy": accuracy,
        "mismatch_data": mismatch_data
    }

# --- PDF Generation Function ---
def generate_pdf_report(doc):
    buffer = BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Document Analysis Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # Metadata
    elements.append(Paragraph("Metadata", styles['Heading2']))
    metadata = [
        f"ID (Filename): {doc.get('id', 'N/A')}",
        f"Source: {doc.get('source', 'N/A')}",
    ]
    if 'data' in doc and isinstance(doc['data'], dict):
        properties = doc['data'].get('properties', {})
        metadata.extend([
            f"Request Timestamp: {properties.get('request_timestamp', 'N/A')}",
            f"Number of Pages: {properties.get('num_pages', 'N/A')}",
            f"Total Time (Seconds): {properties.get('total_time_seconds', 'N/A')}"
        ])
    for line in metadata:
        elements.append(Paragraph(line, styles['Normal']))
        elements.append(Spacer(1, 6))

    # OCR Output
    elements.append(Paragraph("OCR Output", styles['Heading2']))
    ocr_text = "No OCR output available."
    if 'data' in doc and isinstance(doc['data'], dict) and 'extracted_data' in doc['data'] and 'ocr_output' in doc['data']['extracted_data']:
        ocr_text = doc['data']['extracted_data']['ocr_output']
    elements.append(Preformatted(ocr_text, styles['Code'], maxLineLength=80))
    elements.append(Spacer(1, 12))

    # Extracted JSON
    elements.append(Paragraph("Extracted JSON", styles['Heading2']))
    json_text = "No extracted JSON data available."
    if 'data' in doc and isinstance(doc['data'], dict) and 'extracted_data' in doc['data'] and 'gpt_extraction_output' in doc['data']['extracted_data']:
        json_text = json.dumps(doc['data']['extracted_data']['gpt_extraction_output'], indent=2)
    elements.append(Preformatted(json_text, styles['Code'], maxLineLength=80))
    elements.append(Spacer(1, 12))

    pdf.build(elements)
    buffer.seek(0)
    return buffer

# --- Tab Definitions ---
def instructions_tab():
    st.markdown("""
    ## How to Use the Intelligent Data Extraction (JSON) and Model Comparison for PDFs.

    ### Introduction
    The Intelligent Data Extraction (JSON) and Model Comparison for PDFs processes files (PDFs) to extract data using Tesseract OCR, Gemini-2.5-Flash Vision, and Model-3 (Custom Processing). 

    ### Step-by-Step Instructions

    #### 1. Uploading Files
    1. **Navigate to the "🧠 Process Files" tab**.
    2. **Enter a Dataset**:
       - Enter Name of the Dataset.
       - The dataset defines the model prompt and example schema for extraction.
    3. **Configure the Dataset** (Optional):
       - Edit the model prompt or example schema if needed.
       - Click 'Save' to update the configuration in PostgreSQL.
    4. **Choose Processing Model**:
       - Select from Model-1 (PDF to JSON), Model-2 (OCR + Bytes), or Model-3 (Custom Processing).
    5. **Upload Files**:
       - Use the file uploader to select files (PDF, images, etc.).
       - Click 'Submit' to save files to a local folder (`./uploads/<dataset_name>`).
       - Files are processed automatically by the selected model's pipeline.
    6. **What is a Dataset?** 
       - The models process files based on the model prompt and example schema.
       - The schema can be empty; the model will infer a structure if none is provided.

    #### 2. Exploring Data
    1. **Navigate to the "🔎 Explore Data" tab**.
    2. **Fetch Data**:
       - Data is fetched from both local PostgreSQL (Model-1/2) and Model-3 API, and displayed in a table.
    3. **Interact with Data**:
       - Select a file from the dropdown on the left to view its metadata, OCR output, extracted JSON, and download full reports (JSON or PDF) on the right.
       - Filter by source (Local or Model-3) if needed.

    #### 3. JSON Accuracy Analysis
    1. **Navigate to the "📊 JSON Accuracy" tab**.
    2. **Compare Outputs**:
       - Select two documents from the combined database or upload JSON files.
       - View matched/mismatched keys, accuracy, and detailed differences.
    3. **Visualize**:
       - See a pie chart of matched vs. mismatched keys.
       - Review a detailed difference table and code diff view.

    #### 4. Adding New Dataset
    1. **Navigate to the "🧠 Process Files" tab**.
    2. **Add New Dataset**:
       - Enter a dataset name, model prompt, and example schema.
       - Click 'Add New Dataset' to save to PostgreSQL (requires backend support).

    ### Backend Processes
    1. **File Upload**:
       - Files are saved to `./uploads/<dataset_name>` in the backend.
    2. **Processing Pipeline**:
       - **Model-1**: Tesseract performs OCR, and Gemini-2.5-Flash Vision extracts structured JSON.
       - **Model-2**: Processes images in bytes with OCR and passes to Gemini LLM.
       - **Model-3**: Custom processing pipeline (details depend on the specific implementation).
    3. **Data Storage**:
       - Results and metadata are stored in a local PostgreSQL database (Model-1/2) or Model-3 API storage.
    4. **Data Retrieval**:
       - The "Explore Data" tab fetches data from both sources via the backend APIs.

    ### Additional Information
    Source code: [GitHub Repo](#) Available Soon.
    """)

def process_files_tab():
    st.header("🧠 Process Files")

    api_choice = st.selectbox(
        "Choose Processing API or Processing Model",
        options=[
            ("process/pdf2json", "Model-1 PDF to JSON - Pytesseract(OCR) + Gemini LLM [API Used: process/pdf2json]"),
            ("process/OCR+bytes", "Model-2 OCR + Bytes - Images in Bytes to LLM [API Used: process/OCR+bytes]"),
            ("upload", "Model-3 Visdeal & Apicem Invoice Extraction (Pytesseract + Gemini LLM)")
        ],
        format_func=lambda x: x[1]
    )

    dataset_name = st.text_input("Dataset Name", value="")
    model_prompt = st.text_area(
        "Model Prompt",
        value="Extract all data with respect to this schema and give the json values in the same order as this schema."
    )
    example_schema = st.text_area("Example Schema (JSON)", value="{}")

    uploaded_file = st.file_uploader("Upload a file (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"])

    if st.button("Submit"):
        if not dataset_name.strip():
            st.error("Please enter a Dataset Name.")
            return

        if uploaded_file is None:
            st.error("Please upload a file to process.")
            return

        try:
            json.loads(example_schema)
        except json.JSONDecodeError:
            st.error("Example Schema is not a valid JSON.")
            return

        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        data = {
            "dataset_name": dataset_name,
            "model_prompt": model_prompt,
            "example_schema": example_schema
        }

        endpoint = f"/{api_choice[0]}"
        # Use different base URL for Model-3 (when api_choice[0] == "upload")
        # base_url = MODEL3_API_BASE_URL if api_choice[0] == "upload" else "http://localhost:8000"
        base_url = MODEL3_API_BASE_URL if api_choice[0] == "upload" else "https://tuboid-nonparochially-christian.ngrok-free.dev"
        url = base_url + endpoint

        headers = {
            "accept": "application/json"
        }

        with st.spinner("Processing file..."):
            try:
                response = requests.post(url, files=files, data=data, headers=headers)
                response.raise_for_status()
                result = response.json()
                st.success(f"File '{uploaded_file.name}' processed successfully!")
                st.json(result)
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")

def explore_data_tab():
    st.header("🔎 Explore Data")

    # Fetch all documents from both sources
    documents = fetch_all_documents()
    if not documents:
        st.warning("No documents found across all sources.")
        return

    df = pd.DataFrame(documents)
    
    # Ensure 'id' column exists
    if 'id' not in df.columns:
        st.error("No 'id' column found in the data.")
        return
    
    # Extract fields from 'data' column
    if 'data' in df.columns:
        try:
            df['filename'] = df['data'].apply(lambda x: x.get('properties', {}).get('blob_name', 'Unknown') if isinstance(x, dict) else 'Unknown')
            df['request_timestamp'] = df['data'].apply(lambda x: x.get('properties', {}).get('request_timestamp', None) if isinstance(x, dict) else None)
        except Exception as e:
            st.warning(f"Error extracting fields from 'data' column: {str(e)}")

    # Add source column if not present
    if 'source' not in df.columns:
        df['source'] = 'unknown'

    # Filter option for source
    source_filter = st.multiselect(
        "Filter by Source",
        options=df['source'].unique(),
        default=df['source'].unique()
    )
    filtered_df = df[df['source'].isin(source_filter)]

    # Two-column layout
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Select Document")
        selected_doc_id = st.selectbox(
            "Choose a Document",
            options=filtered_df['id'].tolist(),
            format_func=lambda x: f"{x} ({filtered_df[filtered_df['id'] == x]['source'].iloc[0] if not filtered_df.empty else 'unknown'})",
            key="explore_select"
        )

    with col2:
        st.subheader("Document Details")
        if selected_doc_id is not None and not filtered_df.empty:
            doc = filtered_df[filtered_df['id'] == selected_doc_id].iloc[0]
            # Display metadata in a clean table-like format
            st.markdown("**Metadata**")
            metadata = [
                {"Field": "ID (Filename)", "Value": doc['id']},
                {"Field": "Source", "Value": doc.get('source', 'N/A')},
            ]
            if 'data' in doc and isinstance(doc['data'], dict):
                properties = doc['data'].get('properties', {})
                metadata.extend([
                    {"Field": "Request Timestamp", "Value": properties.get('request_timestamp', 'N/A')},
                    {"Field": "Number of Pages", "Value": properties.get('num_pages', 'N/A')},
                    {"Field": "Total Time (Seconds)", "Value": properties.get('total_time_seconds', 'N/A')}
                ])
            st.table(metadata)

            # Display OCR output in an expander
            if 'data' in doc and isinstance(doc['data'], dict) and 'extracted_data' in doc['data'] and 'ocr_output' in doc['data']['extracted_data']:
                with st.expander("View OCR Output", expanded=False):
                    st.code(doc['data']['extracted_data']['ocr_output'], language="text")
            else:
                st.warning("No OCR output available.")

            # Display JSON in an expander
            if 'data' in doc and isinstance(doc['data'], dict) and 'extracted_data' in doc['data'] and 'gpt_extraction_output' in doc['data']['extracted_data']:
                with st.expander("View Extracted JSON", expanded=False):
                    st.code(json.dumps(doc['data']['extracted_data']['gpt_extraction_output'], indent=2), language="json")
                    # Download button for JSON
                    json_str = json.dumps(doc['data']['extracted_data']['gpt_extraction_output'], indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"{doc['id'].replace('/', '_')}_extracted.json",
                        mime="application/json"
                    )
            else:
                st.warning("No extracted JSON data available.")

            # Download button for full analysis report (JSON)
            if 'data' in doc and isinstance(doc['data'], dict):
                analysis_report = {
                    "id": doc['id'],
                    "source": doc.get('source', 'N/A'),
                    "metadata": doc['data'].get('properties', {}),
                    "ocr_output": doc['data'].get('extracted_data', {}).get('ocr_output', None),
                    "gpt_extraction_output": doc['data'].get('extracted_data', {}).get('gpt_extraction_output', None)
                }
                analysis_str = json.dumps(analysis_report, indent=2)
                st.download_button(
                    label="Download Full Analysis Report (JSON)",
                    data=analysis_str,
                    file_name=f"{doc['id'].replace('/', '_')}_analysis_report.json",
                    mime="application/json"
                )

            # Download button for PDF report
            if 'data' in doc and isinstance(doc['data'], dict):
                pdf_buffer = generate_pdf_report(doc)
                st.download_button(
                    label="Download Full Analysis Report (PDF)",
                    data=pdf_buffer,
                    file_name=f"{doc['id'].replace('/', '_')}_analysis_report.pdf",
                    mime="application/pdf"
                )

    # Display table below
    st.subheader("All Documents")
    display_columns = ['id', 'source']
    if 'filename' in df.columns:
        display_columns.append('filename')
    if 'request_timestamp' in df.columns:
        display_columns.append('request_timestamp')
    st.dataframe(filtered_df[display_columns], use_container_width=True)

def json_accuracy_tab():
    st.header("📊 JSON Accuracy")

    # Fetch all documents
    documents = fetch_all_documents()
    if not documents:
        return

    df = pd.DataFrame(documents)
    
    if 'id' not in df.columns:
        st.error("No 'id' column found in the data.")
        return

    st.subheader("Compare JSON Outputs")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Select First JSON**")
        doc1_source = st.radio("Source for First JSON", ["Combined Database", "Upload"], key="doc1_source")
        if doc1_source == "Combined Database":
            doc1_id = st.selectbox(
                "Select First Document",
                options=df['id'].tolist(),
                format_func=lambda x: f"{x} ({df[df['id'] == x]['source'].iloc[0] if not df[df['id'] == x].empty else 'unknown'})",
                key="doc1"
            )
            doc1 = df[df['id'] == doc1_id].iloc[0] if doc1_id is not None else None
            json1 = doc1['data']['extracted_data']['gpt_extraction_output'] if doc1 is not None and 'data' in doc1 and isinstance(doc1['data'], dict) and 'extracted_data' in doc1['data'] and 'gpt_extraction_output' in doc1['data']['extracted_data'] else None
            if json1:
                st.code(json.dumps(json1, indent=2), language="json")
        else:
            uploaded_json1 = st.file_uploader("Upload First JSON", type="json", key="json1")
            if uploaded_json1:
                try:
                    json1 = json.load(uploaded_json1)
                    st.code(json.dumps(json1, indent=2), language="json")
                except json.JSONDecodeError:
                    st.error("Invalid JSON file.")
                    json1 = None
            else:
                json1 = None

    with col2:
        st.write("**Select Second JSON**")
        doc2_source = st.radio("Source for Second JSON", ["Combined Database", "Upload"], key="doc2_source")
        if doc2_source == "Combined Database":
            doc2_id = st.selectbox(
                "Select Second Document",
                options=df['id'].tolist(),
                format_func=lambda x: f"{x} ({df[df['id'] == x]['source'].iloc[0] if not df[df['id'] == x].empty else 'unknown'})",
                key="doc2"
            )
            doc2 = df[df['id'] == doc2_id].iloc[0] if doc2_id is not None else None
            json2 = doc2['data']['extracted_data']['gpt_extraction_output'] if doc2 is not None and 'data' in doc2 and isinstance(doc2['data'], dict) and 'extracted_data' in doc2['data'] and 'gpt_extraction_output' in doc2['data']['extracted_data'] else None
            if json2:
                st.code(json.dumps(json2, indent=2), language="json")
        else:
            uploaded_json2 = st.file_uploader("Upload Second JSON", type="json", key="json2")
            if uploaded_json2:
                try:
                    json2 = json.load(uploaded_json2)
                    st.code(json.dumps(json2, indent=2), language="json")
                except json.JSONDecodeError:
                    st.error("Invalid JSON file.")
                    json2 = None
            else:
                json2 = None

    if json1 and json2:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Comparison Results")
        comparison = compare_json(json1, json2)
        st.success(f"Matched Keys: {comparison['matched_keys']} / {comparison['total_keys']}")
        st.error(f"Mismatched Keys: {comparison['mismatched_keys']}")
        st.info(f"Accuracy: **{comparison['accuracy']:.2f}%**")

        st.subheader("Match vs Mismatch Analysis")
        df_chart = pd.DataFrame({
            'Status': ['Matched', 'Mismatched'],
            'Count': [comparison['matched_keys'], comparison['mismatched_keys']]
        })
        fig = px.pie(df_chart, names='Status', values='Count', color='Status',
                     color_discrete_map={'Matched': 'blue', 'Mismatched': 'brown'},
                     title="JSON Key Match Breakdown")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Detailed JSON Difference")
        old_str = json.dumps(json1, indent=2)
        new_str = json.dumps(json2, indent=2)
        st_code_diff(old_string=old_str, new_string=new_str,
                     language="json", output_format="side-by-side", diff_style="word")

        st.subheader("Mismatched Key Details")
        if comparison['mismatch_data']:
            mismatch_df = pd.DataFrame(comparison['mismatch_data'], columns=["Key", "Expected", "Found"])
            st.dataframe(mismatch_df, use_container_width=True)
        else:
            st.success("No mismatched keys found.")




def main():
    st.header("Intelligent Data Extraction (JSON) and Model Comparison for PDFs")
    tabs = st.tabs(["🧠 Process Files", "🔎 Explore Data", "📊 JSON Accuracy", "🖥️ Instructions"])
    
    with tabs[0]:
        process_files_tab()
    with tabs[1]:
        explore_data_tab()
    with tabs[2]:
        json_accuracy_tab()
    with tabs[3]:
        instructions_tab()

if __name__ == "__main__":
    main()

