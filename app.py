import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objs as go

# Fixed index mapping with correct Yahoo Finance symbols
INDEX_MAP = {
    'NIFTY50': '^NSEI',
    'NIFTYNEXT50': '^NSEMDCP50',
    'NIFTY100': '^CNX100',
    'NIFTY200': '^NSE200',
    'NIFTY500': '^CRSLDX',
    'NIFTYMIDCAP150': 'NIFTY_MIDCAP150.NS',
    'NIFTYSMALLCAP250': 'NIFTY_SMALLCAP250.NS'
}

# Full stock lists (truncated in preview)
STOCK_LISTS = {
    'NIFTY50': ['ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BEL.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'ITC.NS', 'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SHRIRAMFIN.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'TRENT.NS', 'ULTRACEMCO.NS', 'WIPRO.NS'],
    # Other lists truncated for brevity
}

def download_data(symbol, period='1y'):
    """Download historical stock data with proper index handling"""
    try:
        data = yf.download(symbol, period=period, progress=False)
        if data.empty:
            return None
        # Ensure proper datetime index
        data.index = pd.to_datetime(data.index)
        return data
    except Exception:
        return None

def calculate_movement(data):
    """Calculate daily price movement with index reset"""
    if data is None or data.empty:
        return None
    data = data.copy()
    data['Movement'] = np.where(data['Close'] > data['Close'].shift(1), 1, 0)
    return data.dropna().reset_index()

def train_prediction_model(stock_data, index_data):
    """Train prediction model with proper data alignment"""
    if stock_data is None or index_data is None:
        return None, 0
    
    # Merge on date column instead of index
    merged = pd.merge(stock_data[['Date', 'Movement']], 
                     index_data[['Date', 'Movement']], 
                     on='Date', 
                     suffixes=('_stock', '_index'))
    
    if len(merged) < 10:
        return None, 0
    
    X = merged[['Movement_index']]
    y = merged['Movement_stock']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)  # Reduced complexity
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

def main():
    st.set_page_config(
        page_title="IndexSync Predictor",
        page_icon="📈",
        layout="wide"
    )
    st.title("📈 IndexSync Stock Predictor")
    st.markdown("Predict stock movements based on index trends")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        selected_index = st.selectbox("Select Index", list(INDEX_MAP.keys()))
        period = st.selectbox("Data Period", ['3mo', '6mo', '1y', '2y'], index=2)
        st.info("Note: First run may take 30-60 seconds to fetch data")
    
    # Download index data
    index_symbol = INDEX_MAP[selected_index]
    with st.spinner(f"Fetching {selected_index} data..."):
        index_data = download_data(index_symbol, period)
    
    if index_data is None:
        st.error("❌ Failed to download index data. Please try another index or time period.")
        return
    
    # Process index data
    index_data = calculate_movement(index_data)
    if index_data is None:
        st.error("❌ Insufficient index data for analysis")
        return
        
    index_up = index_data['Movement'].mean() * 100
    
    # Display index info
    st.subheader(f"{selected_index} Index Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Index Symbol", index_symbol)
    col2.metric("Historical Up Days", f"{index_up:.2f}%")
    
    # Show index chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=index_data['Date'], 
        y=index_data['Close'], 
        name='Close Price', 
        line=dict(color='#636EFA')
    ))
    fig.update_layout(
        title=f"{selected_index} Price Trend",
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Stock prediction section
    st.subheader("Stock Movement Predictions")
    st.info(f"Analyzing stocks in {selected_index} based on index correlation")
    
    # Get stocks for selected index
    stocks = STOCK_LISTS[selected_index][:20]  # Limit to 20 for performance
    
    # Analyze each stock
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, stock_symbol in enumerate(stocks):
        status_text.text(f"Analyzing {stock_symbol} ({i+1}/{len(stocks)})")
        progress_bar.progress((i+1)/len(stocks))
        
        # Download stock data
        stock_data = download_data(stock_symbol, period)
        if stock_data is None:
            continue
            
        # Process stock data
        stock_data = calculate_movement(stock_data)
        if stock_data is None:
            continue
            
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
            'Stock': stock_symbol.replace('.NS', ''),
            'Prediction': '↑ Up' if prediction == 1 else '↓ Down',
            'Confidence': f"{max(prediction_prob)*100:.1f}%",
            'Actual': '↑ Up' if actual_movement == 1 else '↓ Down',
            'Accuracy': f"{accuracy*100:.1f}%",
            'Correct': 1 if prediction == actual_movement else 0
        })
    
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    if results:
        results_df = pd.DataFrame(results)
        
        # Color formatting
        def color_prediction(val):
            color = 'green' if 'Up' in val else 'red'
            return f'color: {color}'
        
        st.dataframe(
            results_df.style.applymap(color_prediction, subset=['Prediction', 'Actual']),
            use_container_width=True
        )
        
        # Show summary
        correct = sum(results_df['Correct'])
        accuracy = correct / len(results) * 100
        st.metric("Overall Prediction Accuracy", f"{accuracy:.1f}%", 
                 delta_color="off")
    else:
        st.warning("⚠️ No valid predictions generated. Try a different index or time period.")

if __name__ == "__main__":
    main()
