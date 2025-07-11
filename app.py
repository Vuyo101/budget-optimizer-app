import streamlit as st
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Verify TensorFlow version
st.write(f"TensorFlow version: {tf.__version__}")

@st.cache_resource
def load_models():
    try:
        model = tf.keras.models.load_model("tft_model.keras")
        model.load_weights("tft.weights.h5")
        scaler_x = joblib.load("scaler_x.joblib")
        scaler_y = joblib.load("scaler_y.joblib")
        return model, scaler_x, scaler_y
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None, None

def main():
    st.title("National Budget Optimizer")
    
    model, scaler_x, scaler_y = load_models()
    
    if model:
        st.success("✅ Models loaded successfully!")
        
        with st.sidebar:
            st.header("Parameters")
            budget = st.number_input("Total Budget ($)", min_value=1e9, value=5e11, step=1e9)
            scenario = st.selectbox("Economic Scenario", ["Stable", "Recession", "Growth"])
        
        if st.button("Optimize Allocations"):
            with st.spinner("Calculating optimal allocations..."):
                sectors = ["Healthcare", "Education", "Defense", "Infrastructure", 
                          "Agriculture", "Energy", "Environment", "Technology",
                          "Transport", "Public Safety", "Social Welfare"]
                
                # Mock allocations - replace with your actual model predictions
                if scenario == "Recession":
                    alloc = np.array([0.15, 0.2, 0.05, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05])
                elif scenario == "Growth":
                    alloc = np.array([0.1, 0.15, 0.1, 0.15, 0.05, 0.1, 0.05, 0.15, 0.05, 0.05, 0.05])
                else:
                    alloc = np.array([0.12, 0.18, 0.08, 0.12, 0.08, 0.08, 0.06, 0.12, 0.08, 0.06, 0.06])
                
                # Predict GDP impact
                scaled_input = scaler_x.transform(alloc.reshape(1, -1))
                gdp_pred = model.predict(scaled_input)
                gdp_growth = scaler_y.inverse_transform(gdp_pred)[0][0]
                
                # Create results
                allocations = (alloc * budget).astype(int)
                results = pd.DataFrame({
                    "Sector": sectors,
                    "Allocation ($)": allocations,
                    "% of Budget": (alloc * 100).round(2)
                })
                
                # Display results
                st.subheader("Optimal Allocations")
                st.dataframe(results)
                
                # Visualizations
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                results.set_index("Sector")["Allocation ($)"].plot.bar(ax=ax[0])
                ax[0].set_title("Allocation by Sector")
                ax[1].bar(["GDP Growth"], [gdp_growth], color='green')
                ax[1].set_title("Projected GDP Growth")
                st.pyplot(fig)
                
                # Generate report
                report = f"""
                # Budget Allocation Report
                **Date:** {datetime.now().strftime('%Y-%m-%d')}
                **Total Budget:** ${budget:,.2f}
                **Projected GDP Growth:** {gdp_growth:.2f}%
                
                ## Allocation Summary
                {results.to_markdown(index=False)}
                
                ## Recommendations
                - Focus on {results.nlargest(3, 'Allocation ($)')['Sector'].tolist()} sectors
                - Monitor economic indicators quarterly
                - Review allocations every 6 months
                """
                
                st.download_button(
                    "Download Full Report",
                    report,
                    file_name="budget_report.md",
                    mime="text/markdown"
                )

if __name__ == "__main__":
    main()

      
   
    
   
         
      
       

    
  
   
       
