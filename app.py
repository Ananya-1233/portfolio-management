import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
from datetime import datetime, date, timedelta
# Define sectors and stock options (sample data)
st.title("Stock Data Acquisition Tool")
sectors = {
    'Pharmaceuticals': ['DR. REDDY LABORATORIES', 'SUN PHARMACEUTICAL INDUSTRIES', 'CIPLA'],
    'Automotive': ['TATA MOTORS', 'MAHINDRA & MAHINDRA', 'MARUTI SUZUKI INDIA'],
    'Technology': ['TATA CONSULTANCY SERVICES', 'INFOSYS', 'WIPRO', 'HCL TECHNOLOGIES', 'TECH MAHINDRA'],
    'Infrastructure': ['LARSEN & TOUBRO', 'BHARAT HEAVY ELECTRICALS LIMITED', 'POWER GRID CORPORATION OF INDIA', 'RELIANCE INFRASTRUCTURE']
}

# Define ticker symbols mapping
mapping = {
    'DR. REDDY LABORATORIES': 'RDY',
    'LARSEN & TOUBRO': 'LT.BO',
    'TATA MOTORS': 'TATAMOTORS.BO',
    'TATA CONSULTANCY SERVICES': 'TCS.BO',
    'SUN PHARMACEUTICAL INDUSTRIES': 'SUNPHARMA.BO',
    'BHARAT HEAVY ELECTRICALS LIMITED': 'BHEL.BO',
    'MAHINDRA & MAHINDRA': 'M&M.BO',
    'INFOSYS': 'INFY.BO',
    'CIPLA': 'CIPLA.BO',
    'POWER GRID CORPORATION OF INDIA': 'POWERGRID.BO',
    'MARUTI SUZUKI INDIA': 'MARUTI.BO',
    'WIPRO': 'WIPRO.BO',
    'APOLLO HOSPITALS ENTERPRISE': 'APOLLOHOSP.BO',
    'HERO MOTOCORP': 'HEROMOTOCO.BO',
    'HCL TECHNOLOGIES': 'HCLTECH.BO',
    'FORTIS HEALTHCARE': 'FORTIS.BO',
    'RELIANCE INFRASTRUCTURE': 'RELINFRA.BO',
    'BAJAJ AUTO': 'BAJAJ-AUTO.BO',
    'TECH MAHINDRA': 'TECHM.BO'
}

# Define the index ticker
index_ticker = "^BSESN"  # S&P BSE SENSEX

# Define the date range
# start_date = "2024-01-01"
# end_date = "2024-06-30

# start_date = st.date_input("Select a start date")
# end_date = date.today()

weeks_requested = st.number_input("Enter the number of weeks of data needed:", min_value=1, value=10, step=1)

# Calculate the start date based on the number of weeks requested
end_date = date.today()
start_date = end_date - timedelta(weeks=weeks_requested)
st.write(f"Fetching data from {start_date} to {end_date}")

# Create the sector multiselect dropdown
sectors_selected = st.multiselect('Select sectors:', list(sectors.keys()))

# Create the multiselect for selecting stocks based on the selected sectors
selected_options = []
if sectors_selected:
    stock_options = []
    for sector in sectors_selected:
        stock_options += sectors[sector]
    selected_options = st.multiselect('Select stocks:', stock_options)

def calculate_weekly_returns(ticker, stock_name):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.index = pd.to_datetime(data.index)
    weekly_data = data['Adj Close'].resample('W').ffill()
    weekly_return = weekly_data.pct_change() * 100
    result = pd.DataFrame({f'{stock_name}_Weekly Return': weekly_return})
    return result, data

def cal_week(ticker, stock_name):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.index = pd.to_datetime(data.index)
    weekly_data = data['Adj Close'].resample('W').ffill()
    weekly_return = weekly_data.pct_change() * 100
    result = pd.DataFrame({f'{stock_name}_Weekly_Return': weekly_return})
    return result

def highlight_negatives(data, ax, mask):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] < 0:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))

# Add a submit button
if st.button('Calculate weekly returns for selected stocks'):
    if selected_options:
        st.write(f'You chose: {", ".join(selected_options)}')
        
        # Download and display the stock data with weekly returns for each selected stock
        consolidated_data = pd.DataFrame()
        for selected_option in selected_options:
            stock_ticker = mapping[selected_option]
            stock_weekly_returns, _ = calculate_weekly_returns(stock_ticker, selected_option)
            consolidated_data = pd.concat([consolidated_data, stock_weekly_returns], axis=1)

        # Download BSE index data
        index_weekly_returns, _ = calculate_weekly_returns(index_ticker, 'BSE')
        consolidated_data = consolidated_data.join(index_weekly_returns, how='inner')
        consolidated_data = consolidated_data.round(2)
        consolidated_data[f'{selected_option}_Weekly Return'] = consolidated_data[f'{selected_option}_Weekly Return'].apply(
            lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x)
        # Plotting the line graph of all selected stocks' weekly returns in one chart
        fig, ax = plt.subplots(figsize=(10, 6))

        for selected_option in selected_options:
            ax.plot(consolidated_data.index, consolidated_data[f'{selected_option}_Weekly Return'], label=selected_option)

        # Plotting BSE index returns for comparison
        ax.plot(index_weekly_returns.index, index_weekly_returns['BSE_Weekly Return'], label='BSE', linestyle='--', color='black')

        ax.set_title('Weekly Returns of Selected Stocks and BSE')
        ax.set_xlabel('Date')
        ax.set_ylabel('Weekly Return (%)')
        ax.legend()
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        st.pyplot(fig)

        st.write('Consolidated Data')
        st.write(consolidated_data)

        # Provide download link for the consolidated data
        csv = consolidated_data.to_csv().encode('utf-8')
        st.download_button(
            label="Download consolidated data",
            data=csv,
            file_name='consolidated_portfolio_returns.csv',
            mime='text/csv',
        )
         # Plotting the correlation heatmap
        corr = consolidated_data.corr()
        st.write(corr)
    
        cov = consolidated_data.cov()
        st.write(cov)
    
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = corr.values
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax, mask=None)
        highlight_negatives(corr_matrix, ax, None)
        plt.title('Correlation Matrix Heatmap with Highlighted Negative Values')
        st.pyplot(fig)

        # Visualize the covariance matrix using a heatmap with highlighted negative values
        fig, ax = plt.subplots(figsize=(10, 8))
        cov_matrix = cov.values
        sns.heatmap(cov, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax, mask=None)
        highlight_negatives(cov_matrix, ax, None)
        plt.title('Covariance Matrix Heatmap with Highlighted Negative Values')
        st.pyplot(fig)

    else:
        st.write("Please select at least one stock.")

if st.button('Download weekly returns for all stocks'):
    all_weekly_returns = pd.DataFrame()

    for sector, stocks in sectors.items():
        for stock in stocks:
            stock_ticker = mapping[stock]
            stock_weekly_returns = cal_week(stock_ticker, stock)

            if all_weekly_returns.empty:
                all_weekly_returns = stock_weekly_returns
            else:
                all_weekly_returns = all_weekly_returns.join(stock_weekly_returns, how='outer')

    # Download BSE index data
    index_weekly_returns = calculate_weekly_returns(index_ticker, 'BSE')
    all_weekly_returns = all_weekly_returns.join(index_weekly_returns, how='outer')

    all_weekly_returns.columns = [col.replace(' ', '_') for col in all_weekly_returns.columns]

    # Drop columns that are not weekly returns
    weekly_return_cols = [col for col in all_weekly_returns.columns if '_Weekly_Return' in col]
    all_weekly_returns = all_weekly_returns[weekly_return_cols]
    all_weekly_returns.dropna(how='all', inplace=True)
    all_weekly_returns.insert(0, 'Serial_Number', range(1, len(all_weekly_returns) + 1))
    all_weekly_returns = all_weekly_returns.round(2)
    st.write(all_weekly_returns)

    # Provide download link for the consolidated data
    csv = all_weekly_returns.to_csv().encode('utf-8')
    st.download_button(
        label="Download consolidated data for all stocks",
        data=csv,
        file_name='All_Stocks_Weekly_Return.csv',
        mime='text/csv',
    )

    if 'BSE_Weekly_Return' in all_weekly_returns.columns:
        for stock in mapping.keys():
            stock_weekly_return_col = f'{stock}_Weekly_Return'
            if stock_weekly_return_col in all_weekly_returns.columns:
                plt.figure(figsize=(10, 6))
                plt.plot(all_weekly_returns['Serial_Number'], all_weekly_returns[stock_weekly_return_col], label=stock)
                plt.plot(all_weekly_returns['Serial_Number'], all_weekly_returns['BSE_Weekly_Return'], label='BSE')
                plt.xlabel('Week')
                plt.ylabel('Weekly Return')
                plt.title(f'{stock} vs BSE Weekly Returns')
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
                st.pyplot(plt)
                plt.close()
    else:
        st.write("BSE_Weekly_Return column not found in the data.")

def get_stock_metrics(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    
    # Standard Deviation of Returns
    std_dev = hist['Close'].pct_change().std() * 100
    
    # Financial Ratios and Market Metrics
    info = stock.info
    market_cap = info.get('marketCap', 'N/A')
    enterprise_value = info.get('enterpriseValue', 'N/A')
    trailing_pe = info.get('trailingPE', 'N/A')
    forward_pe = info.get('forwardPE', 'N/A')
    peg_ratio = info.get('pegRatio', 'N/A')
    price_sales_ratio = info.get('priceToSalesTrailing12Months', 'N/A')
    price_book_ratio = info.get('priceToBook', 'N/A')
    enterprise_val_rev_ratio = info.get('enterpriseToRevenue', 'N/A')
    enterprise_val_ebitda_ratio = info.get('enterpriseToEbitda', 'N/A')
    beta = info.get('beta', 'N/A')
    metrics = {
        'STD_DEV': std_dev,
        'MARKETCAPITAL_BILLIONS': market_cap / 1e9 if market_cap != 'N/A' else 'N/A',
        'ENTERPRISEVALUE_BILLIONS': enterprise_value / 1e9 if enterprise_value != 'N/A' else 'N/A',
        'TRAILING_PE': trailing_pe,
        'FORWARD_PE': forward_pe,
        'PEG_RATIO_5YRS': peg_ratio,
        'PRICE_SALES_RATIO': price_sales_ratio,
        'PRICE_BOOK_RATIO': price_book_ratio,
        'ENTERPRISE_VAL_REV_RATIO': enterprise_val_rev_ratio,
        'ENTERPRISE_VAL_EBITDA_RATIO': enterprise_val_ebitda_ratio,
        'BETA': beta
    }
    
    return metrics

# Function to get the metrics for all stocks
def get_all_metrics():
    all_metrics = []
    for sector, stocks in sectors.items():
        for stock in stocks:
            ticker = mapping[stock]
            metrics = get_stock_metrics(ticker)
            metrics['STOCK'] = stock
            metrics['INDUSTRY'] = sector
            all_metrics.append(metrics)
    df = pd.DataFrame(all_metrics)
    df_last = df.columns[-2]
    df = df[[df_last] + list(df.columns[:-2])]
    return pd.DataFrame(df)

def plot_all_columns_normal_distribution():
    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Load the data
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:", data.head())

        # Filter numeric columns
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        
        if len(numeric_columns) > 0:
            # Plot the normal distribution for each numeric column
            plt.figure(figsize=(12, 8))
            
            for column in numeric_columns:
                # Extract the data for the column
                column_data = data[column].dropna()

                # Fit a normal distribution to the column data
                mean, std_dev = norm.fit(column_data)

                # Generate a range of values for plotting the normal curve
                x = np.linspace(-10,10,1000)
               # x = np.linspace(column_data.min(), column_data.max(), 100)
                y = norm.pdf(x, mean, std_dev)

                # Plot the normal distribution curve
                plt.plot(x, y, label=f'{column} (μ={mean:.2f}, σ={std_dev:.2f})')
            
            # Customize the plot
            plt.title("Normal Distribution of Numeric Columns")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.legend(loc = 'upper center', title="Columns")
            st.pyplot(plt)
        else:
            st.write("No numeric columns found in the CSV file.")
    else:
        st.write("Please upload a CSV file.")

# Run the function in Streamlit
plot_all_columns_normal_distribution()

# Button to calculate metrics
if st.button('Calculate Stock Metrics'):
    metrics_df = get_all_metrics()
    
    # Formatting numeric values
    for col in metrics_df.columns:
        if metrics_df[col].dtype == 'float64':
            metrics_df[col] = metrics_df[col].apply(lambda x: f'{x:.2f}')
    
    st.write(metrics_df)
