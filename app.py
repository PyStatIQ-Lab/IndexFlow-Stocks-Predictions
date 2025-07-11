import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objs as go

# Index mapping to Yahoo Finance symbols
INDEX_MAP = {
    'NIFTY50': '^NSEI',
    'NIFTYNEXT50': '^NSEMDCP50',  # Nifty Next 50 index
    'NIFTY100': '^CNX100',
    'NIFTY200': '^NSE200',
    'NIFTY500': '^NSE500',
    'NIFTYMIDCAP150': '^NSEMDCP50',  # Using Midcap 50 as proxy
    'NIFTYSMALLCAP250': '^NSEBANK'   # Placeholder, no direct equivalent
}

# Sample stock lists (truncated for brevity)
STOCK_LISTS = {
    'NIFTY50': ['ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS'],
    'NIFTYNEXT50': ['ABB.NS', 'ADANIENSOL.NS', 'ADANIGREEN.NS', 'ADANIPOWER.NS', 'ATGL.NS'],
    'NIFTY100': ['ABB.NS', 'ADANIENSOL.NS', 'ADANIENT.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS'],
    'NIFTY200': ['ABB.NS', 'ACC.NS', 'APLAPOLLO.NS', 'AUBANK.NS', 'ADANIENSOL.NS'],
    'NIFTY500': ['360ONE.NS', '3MINDIA.NS', 'ABB.NS', 'ACC.NS', 'AIAENG.NS'],
    'NIFTYMIDCAP150': ['3MINDIA.NS', 'ACC.NS', 'AIAENG.NS', 'APLAPOLLO.NS', 'AUBANK.NS'],
    'NIFTYSMALLCAP250': ['360ONE.NS', 'AADHARHFC.NS', 'AARTIIND.NS', 'AAVAS.NS', 'ACE.NS']
}

def download_data(symbol, period='1y'):
    """Download historical stock data"""
    try:
        data = yf.download(symbol, period=period, progress=False)
        if data.empty:
            st.warning(f"No data found for {symbol}")
            return None
        return data
    except Exception as e:
        st.error(f"Error downloading {symbol}: {str(e)}")
        return None

def calculate_movement(data):
    """Calculate daily price movement"""
    data['Movement'] = np.where(data['Close'] > data['Close'].shift(1), 1, 0)
    return data.dropna()

def train_prediction_model(stock_data, index_data):
    """Train a prediction model for a stock based on index movement"""
    if stock_data is None or index_data is None:
        return None, 0
    
    # Align dates
    merged = pd.merge(stock_data[['Movement']], index_data[['Movement']], 
                     left_index=True, right_index=True, suffixes=('_stock', '_index'))
    
    if len(merged) < 10:
        return None, 0
    
    X = merged[['Movement_index']]
    y = merged['Movement_stock']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

def main():
    st.title("Stock Movement Predictor")
    st.markdown("Predict stock movements based on index trends")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    selected_index = st.sidebar.selectbox("Select Index", list(INDEX_MAP.keys()))
    period = st.sidebar.selectbox("Data Period", ['3mo', '6mo', '1y', '2y'], index=2)
    
    # Download index data
    index_symbol = INDEX_MAP[selected_index]
    index_data = download_data(index_symbol, period)
    
    if index_data is None:
        st.error("Failed to download index data. Please try another index.")
        return
    
    # Process index data
    index_data = calculate_movement(index_data)
    index_up = index_data['Movement'].mean() * 100
    
    # Display index info
    st.subheader(f"{selected_index} Index Analysis")
    col1, col2 = st.columns(2)
    col1.metric("Index Symbol", index_symbol)
    col2.metric("Up Movement %", f"{index_up:.2f}%")
    
    # Show index chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=index_data.index, y=index_data['Close'], 
                           name='Close Price', line=dict(color='royalblue')))
    fig.update_layout(title=f"{selected_index} Price Trend",
                     xaxis_title='Date',
                     yaxis_title='Price',
                     template='plotly_white')
    st.plotly_chart(fig)
    
    # Stock prediction section
    st.subheader("Stock Movement Prediction")
    st.info(f"Predicting stocks in {selected_index} based on index movement")
    
    # Get stocks for selected index
    stocks = STOCK_LISTS[selected_index]
    
    # Analyze each stock
    results = []
    for stock_symbol in stocks:
        # Download stock data
        stock_data = download_data(stock_symbol, period)
        if stock_data is None:
            continue
            
        # Process stock data
        stock_data = calculate_movement(stock_data)
        
        # Train prediction model
        model, accuracy = train_prediction_model(stock_data, index_data)
        if model is None:
            continue
            
        # Predict based on most recent index movement
        last_index_movement = index_data[['Movement']].iloc[-1].values.reshape(1, -1)
        prediction = model.predict(last_index_movement)[0]
        prediction_prob = model.predict_proba(last_index_movement)[0]
        
        # Get actual movement
        actual_movement = stock_data['Movement'].iloc[-1]
        
        results.append({
            'Stock': stock_symbol,
            'Prediction': 'Up' if prediction == 1 else 'Down',
            'Confidence': f"{max(prediction_prob)*100:.1f}%",
            'Actual': 'Up' if actual_movement == 1 else 'Down',
            'Accuracy': f"{accuracy*100:.1f}%"
        })
    
    # Display results
    if results:
        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.applymap(
            lambda x: 'color: green' if x == 'Up' else 'color: red', 
            subset=['Prediction', 'Actual']
        ))
        
        # Show summary
        correct = sum(1 for r in results if r['Prediction'] == r['Actual'])
        accuracy = correct / len(results) * 100
        st.metric("Overall Prediction Accuracy", f"{accuracy:.1f}%")
    else:
        st.warning("No valid predictions generated. Try a different index or time period.")

if __name__ == "__main__":
    main()
