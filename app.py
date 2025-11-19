import streamlit as st
import pandas as pd
import requests
import json
import os
from streamlit_code_diff import st_code_diff
import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import time

st.set_page_config(page_title="Capture Platform", layout="wide", page_icon="📝")

# API_BASE_URL = "http://127.0.0.1:5800/api"
API_BASE_URL = "https://www.clockchamp.com/api/ocrBytes"
# DATA_API_URL = "http://127.0.0.1:5800/data/"
DATA_API_URL = "https://www.clockchamp.com/api/data/"

# Global Customer Names for Dropdown
CUSTOMER_NAMES = ['allseas', 'visdeal', 'berencourt', 'smeetferrybol', 'corybrothers', 'smeetferryead', 'smeetferryfln', 'Generic']

def fetch_documents():
    url = DATA_API_URL
    headers = {}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json().get("data", [])
        if not data:
            st.warning("No documents found in the database.")
        
        for doc in data:
            doc['source'] = 'Capture Platform - Unified Model'  
            
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch documents from {DATA_API_URL}: {str(e)}")
        return []

# JSON Comparison functions
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

# Report Generation (pdf)  
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
        f"Source: {doc.get('source', 'Unified Model')}",
    ]
    if 'data' in doc and isinstance(doc['data'], dict):
        properties = doc['data'].get('properties', {})
        metadata.extend([
            f"Request Timestamp: {properties.get('processed_at', 'N/A')}",
            f"Number of Pages: {properties.get('num_pages', 'N/A')}",
            f"Total Time (Seconds): {properties.get('processing_time_sec', 'N/A')}"
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

def instructions_tab():
    st.markdown("""
    ## How to Use the (Capture Platform - Unified Model)

    ### Introduction
    The Intelligent Data Extraction (JSON) and Model Comparison for PDFs processes files (PDFs/Images) to extract data using a **Capture Platform - Unified Model** pipeline (e.g. OCR + Gemini LLM), which selects the appropriate extraction schema based on the **Customer Name**.

    ### Step-by-Step Instructions

    #### 1. Uploading Files
    1. **Navigate to the "🧠 Process Files" tab**.
    2. **Enter/Select Customer Name**:
        - Use the radio button to choose between manual entry or selection from the dropdown.
        - This name determines which **extraction schema** the backend will use (e.g., `test` loads `schemas/test.json`).
    3. **Upload Files**:
        - Use the file uploader to select files (PDF, images, etc.).
        - Click **'Submit'** to send the file and the customer name to the backend API (`e.g., api/capturePlatform`).
        - The backend handles the processing, saving, and returns the extracted JSON.
    4. **Output Persistence**:
        - Extracted JSON is now saved and displayed using `st.session_state` and will **not disappear** on a page refresh or interaction with other tabs.

    #### 2. Exploring Data
    1. **Navigate to the "🔎 Explore Data" tab**.
    2. **Fetch Data**:
        - Data is fetched from the database and displayed in a table. All documents will show the **"Capture Platform - Unified Model"** source.
    3. **Interact with Data**:
        - Select a file from the dropdown on the left to view its metadata, OCR output, extracted JSON, and download full reports (JSON or PDF) on the right.

    #### 3. JSON Accuracy Analysis
    1. **Navigate to the "📊 JSON Accuracy" tab**.
    2. **Compare Outputs**:
        - Select two documents from the database or upload JSON files. 
        - This is useful for comparing the model's output against a known 'Ground Truth' JSON **(or)** b/w different outputs of the same file.
    3. **Visualize**:
        - See a pie chart of matched vs. mismatched keys.
        - Review a detailed difference table and code diff view.

    ### Backend Processes
    1. **API Endpoint**:
        - All file processing goes through our endpoint.
    2. **Schema Loading**:
        - The backend uses the `customer_name` to dynamically load the specific extraction schema from `schemas/{customer_name}.json`.
    3. **Processing Pipeline**:
        - The backend runs the unified extraction pipeline (e.g., OCR, Image processing, and Gemini LLM extraction).
    4. **Data Storage**:
        - Results and metadata are stored in our database.

    By: NLD India Software Pvt. Ltd.
    """)

def process_files_tab():
    st.header("🧠 Process Files - Capture Platform (Unified Model)")
    processing_endpoint = "/process/OcrBytes"
    
    customer_input_method = st.radio(
        "**Choose Customer Name Input Method**",
        ("Select from Dropdown", "Enter Customer Name"),
        key="customer_input_method",
        horizontal=True,
        help="Select whether to choose the customer name from a predefined list or manually enter it."
    )

    customer_name = ""
    if customer_input_method == "Enter Customer Name":
        customer_name = st.text_input(
            "Enter Customer Name",
            value="",
            key="manual_customer_name",
            help="This name is used by the backend to load the corresponding extraction schema from 'schemas/{customer_name}.json'."
        )
    else:
        customer_name = st.selectbox(
            "Select Customer Name",
            options=CUSTOMER_NAMES,
            key="dropdown_customer_name",
            help="Choose a customer name from the predefined list: " + ", ".join(CUSTOMER_NAMES)
        )

    st.info(f"The Model Prompt and Example Schema are now loaded by the backend based on the **Customer Name** you enter/select: **{customer_name}**")
    
    # Initialize session state for result storage
    if 'processing_result' not in st.session_state:
        st.session_state.processing_result = None

    uploaded_file = st.file_uploader("Upload a file (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"])

    if st.button("Submit"):
        if not customer_name.strip():
            st.error("🚨 Please enter/select a Customer Name.")
            st.session_state.processing_result = None 
            return

        if uploaded_file is None:
            st.error("🚨 Please upload a file to process.")
            st.session_state.processing_result = None 
            return

        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        data = {
            "customer_name": customer_name,
        }
        
        url = API_BASE_URL + processing_endpoint

        headers = {
            "accept": "application/json"
        }

        st.session_state.processing_result = None 

        with st.spinner(f"Processing file for '{customer_name}'...", show_time=True):
            time.sleep(5) 
            try:
                response = requests.post(url, files=files, data=data, headers=headers)
                response.raise_for_status()
                result = response.json()
                
                st.session_state.processing_result = result

                st.success(f"File '{uploaded_file.name}' processed successfully for customer '{customer_name}'!")
                
                st.rerun() 

            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
                st.session_state.processing_result = None 
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.session_state.processing_result = None 

    if st.session_state.processing_result:
        result = st.session_state.processing_result
        gpt_json = result.get("data", {}).get("extracted_data", {}).get("gpt_extraction_output")

        st.markdown("<hr>", unsafe_allow_html=True)
        
        if gpt_json:
            st.markdown("### 📄 Extracted JSON")
            st.code(json.dumps(gpt_json, indent=2), language="json")
            
            st.download_button(
                label="📥 Download Extracted JSON",
                data=json.dumps(gpt_json, indent=2),
                file_name=f"{uploaded_file.name.replace('.pdf', '')}_{customer_name}_extracted.json" if uploaded_file else f"extracted_{customer_name}.json",
                mime="application/json"
            )
        else:
            st.warning("⚠️ Extracted JSON output not found in the response.")

        st.markdown("### 📄 Processed Document : Summary")
        st.json(result)

def explore_data_tab():
    st.header("🔎 Explore Data")
    documents = fetch_documents()
    if not documents:
        st.warning("No documents found.")
        return

    df = pd.DataFrame(documents)
    
    if 'id' not in df.columns:
        st.error("No 'id' column found in the data.")
        return
    
    if 'data' in df.columns:
        try:
            df['filename'] = df['data'].apply(lambda x: x.get('properties', {}).get('blob_name', 'Unknown') if isinstance(x, dict) else 'Unknown')
            df['request_timestamp'] = df['data'].apply(lambda x: x.get('properties', {}).get('request_timestamp', None) if isinstance(x, dict) else None)
        except Exception as e:
            st.warning(f"Error extracting fields from 'data' column: {str(e)}")

    if 'source' not in df.columns:
        df['source'] = 'Unified Model'

    source_filter = st.multiselect(
        "Filter by Source",
        options=df['source'].unique(),
        default=df['source'].unique()
    )
    filtered_df = df[df['source'].isin(source_filter)]

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Select Document")
        selected_doc_id = st.selectbox(
            "Choose a Document",
            options=filtered_df['id'].tolist(),
            format_func=lambda x: f"{x} ({filtered_df[filtered_df['id'] == x]['source'].iloc[0] if not filtered_df.empty and x in filtered_df['id'].tolist() else 'unknown'})",
            key="explore_select"
        )

    with col2:
        st.subheader("Document Details")
        if selected_doc_id is not None and not filtered_df.empty:
            doc = filtered_df[filtered_df['id'] == selected_doc_id].iloc[0]
            st.markdown("**Metadata**")
            metadata = [
                {"Field": "ID (Filename)", "Value": doc['id']},
                {"Field": "Source", "Value": doc.get('source', 'Unified Model')},
            ]
            if 'data' in doc and isinstance(doc['data'], dict):
                properties = doc['data'].get('properties', {})
                metadata.extend([
                    {"Field": "Request Timestamp", "Value": properties.get('processed_at', 'N/A')},
                    {"Field": "Number of Pages", "Value": properties.get('num_pages', 'N/A')},
                    {"Field": "Total Time (Seconds)", "Value": properties.get('processing_time_sec', 'N/A')}
                ])
            st.table(metadata)

            # Display OCR output  
            if 'data' in doc and isinstance(doc['data'], dict) and 'extracted_data' in doc['data'] and 'ocr_output' in doc['data']['extracted_data']:
                with st.expander("View OCR Output", expanded=False):
                    st.code(doc['data']['extracted_data']['ocr_output'], language="text")
            else:
                st.warning("No OCR output available.")

            # Display JSON
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

            # Download full analysis report (JSON)
            if 'data' in doc and isinstance(doc['data'], dict):
                analysis_report = {
                    "id": doc['id'],
                    "source": doc.get('source', 'Unified Model'),
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

            # Download PDF report
            if 'data' in doc and isinstance(doc['data'], dict):
                pdf_buffer = generate_pdf_report(doc)
                st.download_button(
                    label="Download Full Analysis Report (PDF)",
                    data=pdf_buffer,
                    file_name=f"{doc['id'].replace('/', '_')}_analysis_report.pdf",
                    mime="application/pdf"
                )

    # Display table
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
    documents = fetch_documents()
    if not documents:
        return
    
    df = pd.DataFrame(documents)
    
    if 'id' not in df.columns:
        st.error("No 'id' column found in the data.")
        return

    st.subheader("Compare JSON Outputs")
    col1, col2 = st.columns(2)

    # First JSON  
    with col1:
        st.write("**Select First (1st) JSON**")
        doc1_source = st.radio("Source for First JSON", ["Database Output", "Upload Ground Truth"], key="doc1_source")
        json1 = None
        if doc1_source == "Database Output":
            doc1_id = st.selectbox(
                "Select First Document",
                options=df['id'].tolist(),
                format_func=lambda x: f"{x} ({df[df['id'] == x]['source'].iloc[0] if not df[df['id'] == x].empty and x in df['id'].tolist() else 'unknown'})",
                key="doc1"
            )
            doc1 = df[df['id'] == doc1_id].iloc[0] if doc1_id is not None else None
            json1 = doc1['data']['extracted_data']['gpt_extraction_output'] if doc1 is not None and 'data' in doc1 and isinstance(doc1['data'], dict) and 'extracted_data' in doc1['data'] and 'gpt_extraction_output' in doc1['data']['extracted_data'] else None
            if json1:
                st.code(json.dumps(json1, indent=2), language="json")
        else:
            uploaded_json1 = st.file_uploader("Upload First JSON (Ground Truth)", type="json", key="json1")
            if uploaded_json1:
                try:
                    json1 = json.load(uploaded_json1)
                    st.code(json.dumps(json1, indent=2), language="json")
                except json.JSONDecodeError:
                    st.error("Invalid JSON file.")
                    json1 = None
                
    # Second JSON  
    with col2:
        st.write("**Select Second (2nd) JSON**")
        doc2_source = st.radio("Source for Second JSON", ["Database Output", "Upload Ground Truth"], key="doc2_source")
        json2 = None
        if doc2_source == "Database Output":
            doc2_id = st.selectbox(
                "Select Second Document",
                options=df['id'].tolist(),
                format_func=lambda x: f"{x} ({df[df['id'] == x]['source'].iloc[0] if not df[df['id'] == x].empty and x in df['id'].tolist() else 'unknown'})",
                key="doc2"
            )
            doc2 = df[df['id'] == doc2_id].iloc[0] if doc2_id is not None else None
            json2 = doc2['data']['extracted_data']['gpt_extraction_output'] if doc2 is not None and 'data' in doc2 and isinstance(doc2['data'], dict) and 'extracted_data' in doc2['data'] and 'gpt_extraction_output' in doc2['data']['extracted_data'] else None
            if json2:
                st.code(json.dumps(json2, indent=2), language="json")
        else:
            uploaded_json2 = st.file_uploader("Upload Second JSON (Ground Truth)", type="json", key="json2")
            if uploaded_json2:
                try:
                    json2 = json.load(uploaded_json2)
                    st.code(json.dumps(json2, indent=2), language="json")
                except json.JSONDecodeError:
                    st.error("Invalid JSON file.")
                    json2 = None

    if json1 and json2:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Comparison Results")
        comparison = compare_json(json1, json2)
        st.success(f"Matched Keys: **{comparison['matched_keys']}** / **{comparison['total_keys']}**")
        st.error(f"Mismatched Keys: **{comparison['mismatched_keys']}**")
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
    st.title("Capture Platform - Data Extraction (JSON) & O/P Comparison for PDFs/Images")
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
