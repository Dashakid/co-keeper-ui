"""
CoKeeper - GL Categorization App
Streamlit web application for automatic transaction categorization
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
import requests
import os

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="CoKeeper",
    page_icon="📒",
    layout="wide",
    initial_sidebar_state="auto"
)

# Styling
st.markdown("""
<style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    h1 {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = 'quickbooks'
if 'results' not in st.session_state:
    st.session_state.results = None
if 'trained' not in st.session_state:
    st.session_state.trained = False

# Sidebar
st.sidebar.title("🚀 CoKeeper")
st.sidebar.markdown("Automatic GL Categorization")
st.sidebar.divider()

pages = {
    "📤 Upload & Train": "upload",
    "📊 Results": "results",
    "✅ Review": "review",
    "💾 Export": "export",
    "❓ Help": "help"
}

selected = st.sidebar.radio("Navigation", list(pages.keys()))
page = pages[selected]

st.sidebar.divider()
st.sidebar.info(f"""
**Pipeline**: {st.session_state.pipeline.title()}

**Status**: {'✅ Ready' if st.session_state.trained else '⏳ Not trained'}

**Results**: {f'{len(st.session_state.results)} rows' if st.session_state.results is not None else 'None'}
""")

# ============================================================================
# PAGE: UPLOAD & TRAIN
# ============================================================================

if page == "upload":
    st.title("📤 Upload & Train")
    
    st.markdown("""
    Upload your historical transactions and current expenses. 
    The model learns from historical patterns and automatically categorizes new transactions.
    """)
    
    st.divider()
    
    # Pipeline selector
    st.subheader("Select Data Source")
    pipeline = st.radio(
        "Choose pipeline",
        ["quickbooks", "xero"],
        format_func=lambda x: "QuickBooks" if x == "quickbooks" else "Xero",
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if pipeline != st.session_state.pipeline:
        st.session_state.pipeline = pipeline
        st.session_state.results = None
        st.session_state.trained = False
    
    st.divider()
    
    # File uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Historical Data")
        st.markdown("Your past transactions (already categorized)")
        training_file = st.file_uploader(
            "Upload training CSV",
            type=['csv'],
            key='training_file'
        )
    
    with col2:
        st.subheader("Current Expenses")
        st.markdown("Transactions to categorize")
        prediction_file = st.file_uploader(
            "Upload prediction CSV",
            type=['csv'],
            key='prediction_file'
        )
    
    st.divider()
    
    # Training section
    if training_file and prediction_file:
        st.subheader("Ready to Train")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("🤖 Train Model", type="primary", use_container_width=True):
                with st.spinner("Training model..."):
                    try:
                        # Send training file to backend
                        files = {'file': (training_file.name, training_file, 'text/csv')}
                        endpoint = f"{BACKEND_URL}/train_{'xero' if st.session_state.pipeline == 'xero' else 'qb'}"
                        
                        response = requests.post(endpoint, files=files, timeout=300)
                        
                        if response.status_code == 200:
                            st.success("✅ Model trained successfully!")
                            st.session_state.trained = True
                            
                            # Get predictions
                            with st.spinner("Generating predictions..."):
                                pred_files = {'file': (prediction_file.name, prediction_file, 'text/csv')}
                                pred_endpoint = f"{BACKEND_URL}/predict_{'xero' if st.session_state.pipeline == 'xero' else 'qb'}"
                                
                                pred_response = requests.post(pred_endpoint, files=pred_files, timeout=300)
                                
                                if pred_response.status_code == 200:
                                    pred_data = pred_response.json()
                                    st.session_state.results = pd.DataFrame(pred_data.get('predictions', []))
                                    st.success(f"✅ Generated {len(st.session_state.results)} predictions!")
                                    st.rerun()
                                else:
                                    st.error(f"Prediction failed: {pred_response.status_code}")
                        else:
                            st.error(f"Training failed: {response.status_code}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error(f"❌ Cannot connect to backend at {BACKEND_URL}")
                        st.info("Make sure backend is running: `uvicorn main:app --reload`")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col2:
            st.info("💡 Upload both files and click Train Model to get started")
    
    else:
        st.warning("⚠️ Please upload both training and prediction files")
    
    st.divider()
    
    st.subheader("Tips")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**100+ rows** - Use sufficient historical data")
    with col2:
        st.markdown("**Clean data** - Ensure accurate categorization")
    with col3:
        st.markdown("**Consistent** - Use consistent vendor names")


# ============================================================================
# PAGE: RESULTS
# ============================================================================

elif page == "results":
    st.title("📊 Results & Analysis")
    
    if st.session_state.results is None:
        st.warning("No results yet. Train the model first on the Upload & Train tab.")
    else:
        results = st.session_state.results
        
        # Summary metrics
        st.subheader("Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(results))
        with col2:
            if 'Confidence Score' in results.columns:
                avg_conf = results['Confidence Score'].mean() * 100
                st.metric("Avg Confidence", f"{avg_conf:.1f}%")
            else:
                st.metric("Avg Confidence", "N/A")
        with col3:
            if 'Confidence Tier' in results.columns:
                green = (results['Confidence Tier'] == 'GREEN').sum()
                st.metric("High Confidence", green)
            else:
                st.metric("High Confidence", "N/A")
        with col4:
            if 'Confidence Tier' in results.columns:
                needs_review = (results['Confidence Tier'].isin(['YELLOW', 'RED'])).sum()
                st.metric("Needs Review", needs_review)
            else:
                st.metric("Needs Review", "N/A")
        
        st.divider()
        
        # Data table
        st.subheader("Predictions (First 50 rows)")
        display_df = results.head(50).copy()
        
        # Format confidence as percentage if it exists
        if 'Confidence Score' in display_df.columns:
            display_df['Confidence Score'] = display_df['Confidence Score'].apply(
                lambda x: f"{x*100:.1f}%" if isinstance(x, (int, float)) else x
            )
        
        st.dataframe(display_df, use_container_width=True)


# ============================================================================
# PAGE: REVIEW
# ============================================================================

elif page == "review":
    st.title("✅ Review & Verify")
    
    if st.session_state.results is None:
        st.warning("No results yet. Train the model first on the Upload & Train tab.")
    else:
        results = st.session_state.results
        
        st.markdown("Filter predictions by confidence tier")
        
        if 'Confidence Tier' in results.columns:
            tier = st.selectbox(
                "Select Tier",
                ['GREEN', 'YELLOW', 'RED'],
                format_func=lambda x: f"{'🟢' if x == 'GREEN' else '🟡' if x == 'YELLOW' else '🔴'} {x}"
            )
            
            filtered = results[results['Confidence Tier'] == tier]
            
            st.markdown(f"**{len(filtered)}** predictions in {tier} tier")
            
            display_df = filtered.head(50).copy()
            if 'Confidence Score' in display_df.columns:
                display_df['Confidence Score'] = display_df['Confidence Score'].apply(
                    lambda x: f"{x*100:.1f}%" if isinstance(x, (int, float)) else x
                )
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("Confidence Tier column not found in results")


# ============================================================================
# PAGE: EXPORT
# ============================================================================

elif page == "export":
    st.title("💾 Export Results")
    
    if st.session_state.results is None:
        st.warning("No results yet. Train the model first on the Upload & Train tab.")
    else:
        results = st.session_state.results
        
        st.subheader("Download Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**CSV Format**")
            csv = results.to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV",
                csv,
                f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            st.markdown("**Excel Format**")
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                results.to_excel(writer, sheet_name='Predictions', index=False)
            output.seek(0)
            st.download_button(
                "⬇️ Download Excel",
                output,
                f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        st.divider()
        
        st.subheader("Filter Before Download")
        
        if 'Confidence Tier' in results.columns:
            tiers = st.multiselect("Include tiers", ['GREEN', 'YELLOW', 'RED'], default=['GREEN'])
            filtered = results[results['Confidence Tier'].isin(tiers)]
            st.info(f"📊 {len(filtered)} of {len(results)} predictions match filters")
            
            csv_filtered = filtered.to_csv(index=False)
            st.download_button(
                "⬇️ Download Filtered CSV",
                csv_filtered,
                f"predictions_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True,
                type="primary"
            )


# ============================================================================
# PAGE: HELP
# ============================================================================

elif page == "help":
    st.title("❓ Help & Support")
    
    st.markdown("""
    ### About CoKeeper
    
    CoKeeper automatically categorizes General Ledger transactions using machine learning.
    Upload your historical categorized data and current expenses for instant predictions.
    
    ### Getting Started
    
    1. **Upload Training Data** - Provide historical transactions (already categorized)
    2. **Upload Prediction Data** - Provide transactions that need categorization
    3. **Train Model** - Click the Train button to learn patterns
    4. **Review Results** - Check predictions by confidence tier
    5. **Export** - Download results in CSV or Excel format
    
    ### Confidence Tiers
    
    - **🟢 GREEN** - High confidence (90%+) - Ready to use
    - **🟡 YELLOW** - Medium confidence (70-90%) - Worth reviewing
    - **🔴 RED** - Low confidence (<70%) - Needs manual review
    
    ### Tips for Best Results
    
    - Use 100+ historical transactions for training
    - Ensure training data is accurately categorized
    - Use consistent vendor/account names
    - Update model quarterly with new data
    
    ### File Format
    
    Your CSV files should include these columns:
    - **Date** - Transaction date
    - **Name/Contact** - Vendor or payee name
    - **Account/Related account** - GL account category
    - **Memo/Description** - Transaction details
    
    ### Support
    
    For issues or questions, check the backend logs or contact support.
    Backend URL: `{BACKEND_URL}`
    """)
