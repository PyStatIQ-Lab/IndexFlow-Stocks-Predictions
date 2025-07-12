import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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

# Sample stock lists (for demonstration)
STOCK_LISTS = {
    'NIFTY50': ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS'],
    'NIFTYNEXT50': ['PEL.NS', 'TVSMOTOR.NS', 'ADANIGREEN.NS', 'IRCTC.NS', 'DABUR.NS'],
    'NIFTY100': ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS'],
    'NIFTY200': ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS'],
    'NIFTY500': ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS'],
    'NIFTYMIDCAP150': ['AUBANK.NS', 'DALBHARAT.NS', 'PERSISTENT.NS', 'ABCAPITAL.NS', 'IDFCFIRSTB.NS'],
    'NIFTYSMALLCAP250': ['CARBORUNIV.NS', 'KSB.NS', 'MAZDOCK.NS', 'JKPAPER.NS', 'RBLBANK.NS']
}

def download_data(symbol, period='1y'):
    """Download historical stock data with robust error handling"""
    try:
        end_date = datetime.today()
        
        if period == '3mo':
            start_date = end_date - timedelta(days=90)
        elif period == '6mo':
            start_date = end_date - timedelta(days=180)
        elif period == '1y':
            start_date = end_date - timedelta(days=365)
        elif period == '2y':
            start_date = end_date - timedelta(days=730)
        else:
            start_date = end_date - timedelta(days=365)
            
        data = yf.download(
            symbol, 
            start=start_date, 
            end=end_date,
            progress=False, 
            auto_adjust=True
        )
        
        if data.empty:
            st.warning(f"No data found for {symbol}")
            return None
            
        return data
    except Exception as e:
        st.error(f"Error downloading {symbol}: {str(e)}")
        return None

def calculate_returns(data):
    """Calculate daily returns with validation"""
    if data is None or data.empty:
        return None
        
    try:
        data = data.copy()
        if 'Close' in data.columns:
            # Keep all columns, just add returns
            data['Return'] = data['Close'].pct_change()
            data = data.dropna()
            return data
        return None
    except Exception as e:
        st.error(f"Error calculating returns: {str(e)}")
        return None

def calculate_correlation(stock_returns, index_returns):
    """Calculate correlation with detailed error handling"""
    if stock_returns is None or index_returns is None:
        return None, None, None, "Missing returns data"
    
    try:
        # Align datasets using index (datetime)
        merged = pd.merge(
            stock_returns[['Return']],
            index_returns[['Return']],
            left_index=True,
            right_index=True,
            how='inner',
            suffixes=('_stock', '_index')
        )
        
        if len(merged) < 10:
            return None, None, None, f"Only {len(merged)} common trading days"
            
        # Extract columns using position to avoid naming issues
        stock_col = merged.iloc[:, 0]
        index_col = merged.iloc[:, 1]
        
        correlation = stock_col.corr(index_col)
        
        # Prepare data for linear regression
        X = index_col.values.reshape(-1, 1)
        y = stock_col.values
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        beta = model.coef_[0]
        alpha = model.intercept_
        
        return correlation, beta, alpha, "Success"
        
    except Exception as e:
        return None, None, None, f"Calculation error: {str(e)}"

def predict_stock_return(index_percent_change, beta, alpha):
    """Predict stock return based on index change"""
    try:
        index_change_decimal = index_percent_change / 100.0
        predicted_return = (alpha + beta * index_change_decimal) * 100
        return predicted_return
    except:
        return None

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
    
    # Calculate index returns - preserve all columns
    index_returns = calculate_returns(index_data)
    if index_returns is None or index_returns.empty:
        st.error("❌ Insufficient index data for analysis")
        return
    
    # Display index info
    st.subheader(f"{selected_index} Index Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Index Symbol", index_symbol)
    col2.metric("Data Points", len(index_returns))
    
    # FIX: Get last close from original data, not returns data
    last_close = index_data['Close'].iloc[-1] if 'Close' in index_data.columns else "N/A"
    col3.metric("Last Close", f"₹{last_close:.2f}" if isinstance(last_close, float) else last_close)
    
    # Show index chart using original data
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=index_data.index, 
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
    
    # Show index returns distribution
    st.subheader("Index Returns Distribution")
    if 'Return' in index_returns.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(index_returns['Return'] * 100, kde=True, ax=ax, bins=30)
        plt.xlabel('Daily Return (%)')
        plt.title('Distribution of Index Daily Returns')
        st.pyplot(fig)
    else:
        st.warning("Return data not available for index")
    
    # Stock prediction section
    st.subheader("Stock Movement Predictions")
    st.info(f"Analyzing stocks in {selected_index} based on index correlation")
    
    # Get stocks for selected index
    stocks = STOCK_LISTS[selected_index]
    results = []
    debug_info = []
    
    for stock_symbol in stocks:
        # Download stock data
        stock_data = download_data(stock_symbol, period)
        if stock_data is None:
            debug_info.append(f"{stock_symbol}: Download failed")
            continue
            
        # Calculate stock returns
        stock_returns = calculate_returns(stock_data)
        if stock_returns is None or stock_returns.empty:
            debug_info.append(f"{stock_symbol}: Returns calculation failed")
            continue
            
        # Calculate correlation and beta
        correlation, beta, alpha, status = calculate_correlation(stock_returns, index_returns)
        
        if correlation is None:
            debug_info.append(f"{stock_symbol}: {status}")
            continue
            
        # Predict returns for different scenarios
        pred_1 = predict_stock_return(index_change_1, beta, alpha)
        pred_2 = predict_stock_return(index_change_2, beta, alpha)
        pred_3 = predict_stock_return(index_change_3, beta, alpha)
        
        if pred_1 is None:
            debug_info.append(f"{stock_symbol}: Prediction failed")
            continue
            
        results.append({
            'Stock': stock_symbol.replace('.NS', ''),
            'Correlation': f"{correlation:.4f}",
            'Beta': f"{beta:.4f}",
            'Alpha': f"{alpha*100:.4f}%",
            f'If Index +{index_change_1}%': f"{pred_1:.2f}%",
            f'If Index +{index_change_2}%': f"{pred_2:.2f}%",
            f'If Index +{index_change_3}%': f"{pred_3:.2f}%",
        })
    
    # Display results
    if results:
        results_df = pd.DataFrame(results)
        
        # Sort by correlation (highest first)
        results_df['Correlation_Value'] = results_df['Correlation'].astype(float)
        results_df = results_df.sort_values('Correlation_Value', ascending=False)
        results_df = results_df.drop(columns='Correlation_Value')
        
        # Display table
        st.dataframe(results_df, use_container_width=True)
        
        # Show top correlated stock visualization
        if len(results) > 0:
            top_stock = results[0]
            st.subheader(f"Top Correlated Stock: {top_stock['Stock']}")
            
            # Create prediction visualization
            index_changes = np.linspace(-5, 5, 21)
            pred_returns = [
                predict_stock_return(
                    change, 
                    float(top_stock['Beta']), 
                    float(top_stock['Alpha'].replace('%', '')) / 100
                ) 
                for change in index_changes
            ]
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(index_changes, pred_returns, 'b-')
            ax.set_xlabel('Index Change (%)')
            ax.set_ylabel('Predicted Stock Change (%)')
            ax.set_title(f"Prediction Model: {top_stock['Stock']} vs {selected_index}")
            ax.grid(True)
            st.pyplot(fig)
    else:
        st.warning("⚠️ No valid predictions generated")
        
        # Show debug information
        with st.expander("Show Debug Information"):
            st.write("Reasons for failure per stock:")
            for info in debug_info:
                st.write(f"- {info}")
                
            st.write("\n**Common Solutions:**")
            st.write("1. Try a longer time period (1Y or 2Y)")
            st.write("2. Check if symbols are valid (e.g., RELIANCE.NS)")
            st.write("3. Try a different index")
            st.write("4. Check your internet connection")
            st.write("5. Ensure symbols have trading history for selected period")

if __name__ == "__main__":
    main()
