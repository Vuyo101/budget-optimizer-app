import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np

# Custom CSS for styling
st.markdown("""
<style>
    .big-font { font-size:20px !important; }
    .reportview-container { background: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all required models and scalers"""
    try:
        # Load FinBERT
        finbert = AutoModelForSequenceClassification.from_pretrained("models/finbert")
        
        # Load TFT Model
        tft_model = tf.keras.models.load_model("models/tft_model.keras")
        
        # Load Scaler
        scaler = joblib.load("scalers/scaler_x.joblib")
        
        return finbert, tft_model, scaler
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None, None, None

def generate_allocation_report(budget, allocations):
    """Generate formatted report"""
    report = f"""
    # Budget Allocation Report
    **Total Budget:** ${budget:,.2f}
    
    ## Sector Allocations:
    {allocations.to_markdown()}
    
    *Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*
    """
    return report

# --- APP UI ---
st.title("📊 National Budget Optimizer")
st.markdown('<p class="big-font">AI-powered fiscal allocation system</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Parameters")
    budget = st.number_input("Total Budget ($)", min_value=1e9, value=5e11, step=1e9)
    shock_scenario = st.selectbox("Economic Scenario", ["Stable", "Recession", "Growth"])

if st.button("🚀 Optimize Allocations"):
    finbert, model, scaler = load_models()
    
    if model:
        with st.spinner("⚙️ Calculating optimal allocations..."):
            # Simulated optimization (replace with your actual PSO code)
            sectors = ["Healthcare", "Education", "Defense", "Infrastructure", 
                      "Agriculture", "Energy", "Environment", "Technology",
                      "Transport", "Public Safety", "Social Welfare"]
            
            # Mock allocations (replace with model predictions)
            if shock_scenario == "Recession":
                allocations = np.array([0.15, 0.2, 0.05, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05])
            else:
                allocations = np.array([0.1, 0.15, 0.1, 0.15, 0.05, 0.1, 0.05, 0.15, 0.05, 0.05, 0.05])
            
            allocations = (allocations * budget).astype(int)
            results = pd.DataFrame({
                "Sector": sectors,
                "Allocation": allocations,
                "% of Budget": (allocations / budget * 100).round(2)
            })

        # Display Results
        st.success("✅ Optimization complete!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Allocation by Sector")
            st.bar_chart(results.set_index("Sector")["Allocation"])
        
        with col2:
            st.subheader("Budget Distribution")
            st.pyplot(results.plot.pie(
                y="Allocation", 
                labels=results["Sector"],
                autopct="%1.1f%%"
            ).figure)
        
        # Download Report
        report = generate_allocation_report(budget, results)
        st.download_button(
            label="📥 Download Full Report",
            data=report,
            file_name="budget_allocation_report.md",
            mime="text/markdown"
        )