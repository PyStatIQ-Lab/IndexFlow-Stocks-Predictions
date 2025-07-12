import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

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

# Full stock lists
STOCK_LISTS = {
    'NIFTY50': ['ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BEL.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'ITC.NS', 'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SHRIRAMFIN.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'TRENT.NS', 'ULTRACEMCO.NS', 'WIPRO.NS'],
    'NIFTYNEXT50': ['ABB.NS', 'ADANIENSOL.NS', 'ADANIGREEN.NS', 'ADANIPOWER.NS', 'ATGL.NS', 'AMBUJACEM.NS', 'DMART.NS', 'BAJAJHLDNG.NS', 'BANKBARODA.NS', 'BHEL.NS', 'BOSCHLTD.NS', 'CANBK.NS', 'CHOLAFIN.NS', 'DLF.NS', 'DABUR.NS', 'DIVISLAB.NS', 'GAIL.NS', 'GODREJCP.NS', 'HAVELLS.NS', 'HAL.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'IOC.NS', 'IRCTC.NS', 'IRFC.NS', 'NAUKRI.NS', 'INDIGO.NS', 'JSWENERGY.NS', 'JINDALSTEL.NS', 'JIOFIN.NS', 'LTIM.NS', 'LICI.NS', 'LODHA.NS', 'NHPC.NS', 'PIDILITIND.NS', 'PFC.NS', 'PNB.NS', 'RECLTD.NS', 'MOTHERSON.NS', 'SHREECEM.NS', 'SIEMENS.NS', 'TVSMOTOR.NS', 'TATAPOWER.NS', 'TORNTPHARM.NS', 'UNIONBANK.NS', 'UNITDSPR.NS', 'VBL.NS', 'VEDL.NS', 'ZOMATO.NS', 'ZYDUSLIFE.NS'],
    'NIFTY100': ['ABB.NS', 'ADANIENSOL.NS', 'ADANIENT.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS', 'ADANIPOWER.NS', 'ATGL.NS', 'AMBUJACEM.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'DMART.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BAJAJHLDNG.NS', 'BANKBARODA.NS', 'BEL.NS', 'BHEL.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BOSCHLTD.NS', 'BRITANNIA.NS', 'CANBK.NS', 'CHOLAFIN.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DLF.NS', 'DABUR.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GAIL.NS', 'GODREJCP.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HAVELLS.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HAL.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'ITC.NS', 'IOC.NS', 'IRCTC.NS', 'IRFC.NS', 'INDUSINDBK.NS', 'NAUKRI.NS', 'INFY.NS', 'INDIGO.NS', 'JSWENERGY.NS', 'JSWSTEEL.NS', 'JINDALSTEL.NS', 'JIOFIN.NS', 'KOTAKBANK.NS', 'LTIM.NS', 'LT.NS', 'LICI.NS', 'LODHA.NS', 'M&M.NS', 'MARUTI.NS', 'NHPC.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'PIDILITIND.NS', 'PFC.NS', 'POWERGRID.NS', 'PNB.NS', 'RECLTD.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'MOTHERSON.NS', 'SHREECEM.NS', 'SHRIRAMFIN.NS', 'SIEMENS.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TVSMOTOR.NS', 'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATAPOWER.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'TORNTPHARM.NS', 'TRENT.NS', 'ULTRACEMCO.NS', 'UNIONBANK.NS', 'UNITDSPR.NS', 'VBL.NS', 'VEDL.NS', 'WIPRO.NS', 'ZOMATO.NS', 'ZYDUSLIFE.NS'],
    # Other lists truncated for brevity
}

def download_data(symbol, period='1y'):
    """Download historical stock data"""
    try:
        data = yf.download(symbol, period=period, progress=False)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error downloading {symbol}: {str(e)}")
        return None

def calculate_returns(data):
    """Calculate daily returns"""
    if data is None or data.empty:
        return None
    data = data.copy()
    if 'Close' in data.columns:
        data['Return'] = data['Close'].pct_change()
        return data.dropna()
    return None

def calculate_correlation(stock_returns, index_returns):
    """Calculate correlation between stock and index returns"""
    if stock_returns is None or index_returns is None:
        return None, None, None
    
    # Align dates
    merged = pd.merge(
        stock_returns[['Return']], 
        index_returns[['Return']], 
        left_index=True, 
        right_index=True, 
        suffixes=('_stock', '_index')
    )
    
    if len(merged) < 10:
        return None, None, None
    
    # Calculate correlation
    correlation = merged['Return_stock'].corr(merged['Return_index'])
    
    # Prepare data for linear regression
    X = merged[['Return_index']].values
    y = merged['Return_stock'].values
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    beta = model.coef_[0]
    alpha = model.intercept_
    
    return correlation, beta, alpha

def predict_stock_return(index_percent_change, beta, alpha):
    """Predict stock return based on index change"""
    index_change_decimal = index_percent_change / 100.0
    predicted_return = (alpha + beta * index_change_decimal) * 100
    return predicted_return

def main():
    st.set_page_config(
        page_title="IndexSync Predictor",
        page_icon="📈",
        layout="wide"
    )
    st.title("📈 IndexSync Stock Predictor")
    st.markdown("Predict stock movements based on index correlations")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        selected_index = st.selectbox("Select Index", list(INDEX_MAP.keys()))
        period = st.selectbox("Data Period", ['3mo', '6mo', '1y', '2y'], index=2)
        st.info("Note: First run may take 30-60 seconds to fetch data")
        
        st.header("Prediction Parameters")
        index_change_1 = st.number_input("Index Change Scenario 1 (%)", value=1.0, step=0.5)
        index_change_2 = st.number_input("Index Change Scenario 2 (%)", value=2.0, step=0.5)
        index_change_3 = st.number_input("Index Change Scenario 3 (%)", value=5.0, step=0.5)
    
    # Download index data
    index_symbol = INDEX_MAP[selected_index]
    with st.spinner(f"Fetching {selected_index} data..."):
        index_data = download_data(index_symbol, period)
    
    if index_data is None:
        st.error("❌ Failed to download index data. Please try another index or time period.")
        return
    
    # Calculate index returns
    index_returns = calculate_returns(index_data)
    if index_returns is None or index_returns.empty:
        st.error("❌ Insufficient index data for analysis")
        return
    
    # Display index info
    st.subheader(f"{selected_index} Index Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Index Symbol", index_symbol)
    col2.metric("Average Daily Return", f"{index_returns['Return'].mean() * 100:.4f}%")
    col3.metric("Volatility", f"{index_returns['Return'].std() * 100:.4f}%")
    
    # Show index chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=index_returns.index, 
        y=index_returns['Close'], 
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
    
    # Show index returns distribution
    st.subheader("Index Returns Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(index_returns['Return'] * 100, kde=True, ax=ax)
    plt.xlabel('Daily Return (%)')
    plt.title('Distribution of Index Daily Returns')
    st.pyplot(fig)
    
    # Stock prediction section
    st.subheader("Stock Movement Predictions")
    st.info(f"Predicting stock movements in {selected_index} based on index correlations")
    
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
            
        # Calculate stock returns
        stock_returns = calculate_returns(stock_data)
        if stock_returns is None or stock_returns.empty:
            continue
            
        # Calculate correlation and beta
        correlation, beta, alpha = calculate_correlation(stock_returns, index_returns)
        if correlation is None or beta is None:
            continue
            
        # Predict returns for different scenarios
        pred_1 = predict_stock_return(index_change_1, beta, alpha)
        pred_2 = predict_stock_return(index_change_2, beta, alpha)
        pred_3 = predict_stock_return(index_change_3, beta, alpha)
        
        results.append({
            'Stock': stock_symbol.replace('.NS', ''),
            'Correlation': correlation,
            'Beta': beta,
            'Alpha (%)': alpha * 100,
            f'Pred @ {index_change_1}%': f"{pred_1:.2f}%",
            f'Pred @ {index_change_2}%': f"{pred_2:.2f}%",
            f'Pred @ {index_change_3}%': f"{pred_3:.2f}%",
        })
    
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    if results:
        results_df = pd.DataFrame(results)
        
        # Sort by correlation (highest first)
        results_df = results_df.sort_values('Correlation', ascending=False)
        
        # Format correlation and beta
        results_df['Correlation'] = results_df['Correlation'].apply(lambda x: f"{x:.4f}")
        results_df['Beta'] = results_df['Beta'].apply(lambda x: f"{x:.4f}")
        results_df['Alpha (%)'] = results_df['Alpha (%)'].apply(lambda x: f"{x:.4f}%")
        
        # Display table
        st.dataframe(results_df, use_container_width=True)
        
        # Show top correlated stocks
        st.subheader("Top Correlated Stocks")
        top_stocks = results_df.head(5)
        
        for _, row in top_stocks.iterrows():
            stock = row['Stock']
            beta = float(row['Beta'])
            correlation = float(row['Correlation'])
            
            st.markdown(f"**{stock}** (Correlation: {correlation:.4f}, Beta: {beta:.4f})")
            
            # Create predictions for visualization
            index_changes = np.linspace(-5, 5, 21)  # -5% to +5%
            pred_returns = [predict_stock_return(change, beta, float(row['Alpha (%)'].replace('%', '')) / 100) for change in index_changes]
            
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(index_changes, pred_returns, 'b-')
            ax.set_xlabel('Index Change (%)')
            ax.set_ylabel('Predicted Stock Change (%)')
            ax.set_title(f'{stock} Prediction Model')
            ax.grid(True)
            st.pyplot(fig)
            
            # Show actual vs predicted
            st.markdown(f"**Predicted Change for {stock}:**")
            col1, col2, col3 = st.columns(3)
            col1.metric(f"If Index +{index_change_1}%", f"{float(row[f'Pred @ {index_change_1}%'].replace('%', '')):.2f}%")
            col2.metric(f"If Index +{index_change_2}%", f"{float(row[f'Pred @ {index_change_2}%'].replace('%', '')):.2f}%")
            col3.metric(f"If Index +{index_change_3}%", f"{float(row[f'Pred @ {index_change_3}%'].replace('%', '')):.2f}%")
            st.divider()
            
    else:
        st.warning("⚠️ No valid predictions generated. Try a different index or time period.")

if __name__ == "__main__":
    main()
