import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIG ---
st.set_page_config(page_title="SharkPredict AI Pro", layout="wide")

@st.cache_resource
def load_assets():
    model = pickle.load(open('shark_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    cols = pickle.load(open('feature_names.pkl', 'rb'))
    return model, scaler, cols

model, scaler, feature_names = load_assets()

# --- NEW: AI ADVISOR LOGIC FUNCTION ---
def get_shark_tips(prob, ask, equity, valuation, is_profitable):
    tips = []
    # Valuation Logic
    if valuation > 2000000:
        tips.append("🚩 **Valuation Alert:** Your valuation is high ($2M+). Sharks often find this risky for early-stage startups. Consider lowering your 'Ask' to entice them.")
    # Equity Logic
    if equity < 10:
        tips.append("⚖️ **Equity Warning:** You are offering less than 10%. Sharks usually want a larger 'chunk' (20-25%) to stay motivated to help you grow.")
    # Profitability
    if is_profitable == "No":
        tips.append("💰 **Cash Flow Tip:** Since you aren't profitable, focus your pitch on 'User Growth' or 'Proprietary Technology'.")
    else:
        tips.append("✅ **Profitability:** Being profitable is your strongest weapon! Lead your pitch with your revenue numbers.")
    # General Advice
    if prob < 0.5:
        tips.append("💡 **Strategy:** Based on historical data, pitches like yours struggle when the 'Ask' is too high relative to current viewership. Try asking for 20% less cash.")
    return tips

# --- UI DESIGN ---
st.title("🦈 SharkPredict AI: Expert Edition")
st.markdown("---")

# Sidebar for Inputs
st.sidebar.header("Step 1: Financials")
ask = st.sidebar.number_input("Ask Amount ($)", value=100000, step=10000)
equity = st.sidebar.slider("Equity Offered (%)", 1.0, 100.0, 10.0)
viewers = st.sidebar.number_input("US Viewership (Millions)", 1.0, 10.0, 4.5)

st.sidebar.header("Step 2: Business Health")
is_profitable = st.sidebar.radio("Are you Profitable?", ["No", "Yes"])
is_profitable_val = 1 if is_profitable == "Yes" else 0

season = st.sidebar.slider("Season", 1, 18, 17)
multiple = st.sidebar.radio("Multiple Entrepreneurs?", [0, 1])

industries = [c.replace('Industry_', '') for c in feature_names if 'Industry_' in c]
selected_industry = st.sidebar.selectbox("Industry Category", industries)

# --- PREDICTION LOGIC ---
if st.sidebar.button("Analyze Pitch"):
    # Math logic
    valuation_req = ask / (equity / 100)
    val_ratio = ask / (equity + 1)
    view_per_ask = viewers / (ask + 1)
    
    # Build input row
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    input_df['Season Number'] = season
    input_df['Multiple Entrepreneurs'] = multiple
    input_df['US Viewership'] = viewers
    input_df['Original Ask Amount'] = ask
    input_df['Original Offered Equity'] = equity
    input_df['Valuation Requested'] = valuation_req
    input_df['valuation_ratio'] = val_ratio
    input_df['view_per_ask'] = view_per_ask
    input_df['Is_Profitable'] = is_profitable_val 
    input_df[f'Industry_{selected_industry}'] = 1
    
    # Scale and Predict
    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0][1]
    
    # --- DISPLAY ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Shark Verdict")
        if prob > 0.5:
            st.success(f"💰 DEAL! Confidence: {prob:.2%}")
        else:
            st.error(f"🚫 I'M OUT! Rejection Risk: {(1-prob):.2%}")
            
    with col2:
        st.subheader("Why this result?")
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(8)
        fig, ax = plt.subplots()
        sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette='magma', legend=False)
        st.pyplot(fig)

    # --- PASTE THE AI ADVISOR HERE (Bottom of the block) ---
    st.markdown("---")
    st.subheader("🤖 AI Advisor: Pitch Improvement Tips")
    
    # Get tips based on the data entered
    pitch_tips = get_shark_tips(prob, ask, equity, valuation_req, is_profitable)
    
    # Display each tip as an info box
    for tip in pitch_tips:
        st.info(tip)