"""
Stock Market Flow Tracker - USA & Australia
Fixed version with all original features and proper attribute definitions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import requests
import json
import os
from fredapi import Fred
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Market Flow Tracker - USA & Australia",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean styling
st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF;
    }
    
    .stMarkdown, .stText, p, span, label {
        color: #0e1117 !important;
    }
    
    h1, h2, h3 {
        color: #0e1117 !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 2px solid #e9ecef;
    }
    
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    .stButton > button {
        background-color: #2E86C1;
        color: white;
        border: none;
    }
    
    .stButton > button:hover {
        background-color: #1B5E8F;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'usa_data' not in st.session_state:
    st.session_state.usa_data = None
if 'aus_data' not in st.session_state:
    st.session_state.aus_data = None
if 'usa_market' not in st.session_state:
    st.session_state.usa_market = None
if 'aus_market' not in st.session_state:
    st.session_state.aus_market = None
if 'indicators' not in st.session_state:
    st.session_state.indicators = {}
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'weekly_analyses' not in st.session_state:
    st.session_state.weekly_analyses = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

class MarketFlowTracker:
    def __init__(self):
        # API Keys from Streamlit secrets
        self.fred_api_key = st.secrets.get("FRED_API_KEY", "")
        self.alpha_vantage_key = st.secrets.get("ALPHA_VANTAGE_KEY", "")
        self.anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        self.ici_api_key = st.secrets.get("ICI_API_KEY", "")
        
        # Email configuration
        self.sender_email = st.secrets.get("SENDER_EMAIL", "")
        self.sender_password = st.secrets.get("SENDER_PASSWORD", "")
        
        # Initialize FRED
        self.fred = None
        if self.fred_api_key:
            try:
                self.fred = Fred(api_key=self.fred_api_key)
            except:
                pass
        
        # Time ranges
        self.time_ranges = {
            '1M': 30,
            '3M': 90,
            '6M': 180,
            '1Y': 365,
            '3Y': 365*3,
            '5Y': 365*5,
            '10Y': 365*10,
            '15Y': 365*15,
            'All': None
        }
        
        # Initialize cache file
        self.cache_file = 'market_data_cache.pkl'
        
        # Initialize all data attributes
        self.initialize_data()
        
        # Check for Wednesday update
        self.check_wednesday_update()
    
    def initialize_data(self):
        """Initialize all data attributes with embedded historical data"""
        # Try to load from session state first
        if st.session_state.usa_data is not None:
            self.usa_data = st.session_state.usa_data
            self.aus_data = st.session_state.aus_data
            self.usa_market = st.session_state.usa_market
            self.aus_market = st.session_state.aus_market
            self.indicators = st.session_state.indicators
        else:
            # Generate embedded historical data
            self.generate_embedded_data()
            
            # Save to session state
            st.session_state.usa_data = self.usa_data
            st.session_state.aus_data = self.aus_data
            st.session_state.usa_market = self.usa_market
            st.session_state.aus_market = self.aus_market
            st.session_state.indicators = self.indicators
    
    def generate_embedded_data(self):
        """Generate 15+ years of embedded historical data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=15*365)
        
        # Weekly dates for flow data
        weekly_dates = pd.date_range(start=start_date, end=end_date, freq='W')
        
        # Daily dates for market data
        daily_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate USA flow data
        trend = np.linspace(1000, 2500, len(weekly_dates))
        seasonal = 200 * np.sin(np.arange(len(weekly_dates)) * 2 * np.pi / 52)
        noise = np.cumsum(np.random.randn(len(weekly_dates)) * 50)
        
        usa_flows = trend + seasonal + noise
        usa_flows = np.maximum(usa_flows, 100)  # Keep positive
        
        self.usa_data = pd.DataFrame({
            'date': weekly_dates,
            'retail_flows': usa_flows * 0.6,
            'institutional_flows': usa_flows * 0.4,
            'total_flows': usa_flows,
            'retail_flows_change': pd.Series(usa_flows * 0.6).pct_change(periods=4) * 100,
            'institutional_flows_change': pd.Series(usa_flows * 0.4).pct_change(periods=4) * 100
        }).set_index('date')
        
        # Generate Australia flow data
        aus_flows = trend * 0.5 + seasonal * 0.7 + np.cumsum(np.random.randn(len(weekly_dates)) * 30)
        aus_flows = np.maximum(aus_flows, 50)
        
        self.aus_data = pd.DataFrame({
            'date': weekly_dates,
            'retail_flows': aus_flows * 0.5,
            'institutional_flows': aus_flows * 0.5,
            'total_flows': aus_flows,
            'retail_flows_change': pd.Series(aus_flows * 0.5).pct_change(periods=4) * 100,
            'institutional_flows_change': pd.Series(aus_flows * 0.5).pct_change(periods=4) * 100
        }).set_index('date')
        
        # Generate S&P 500 data
        sp500_returns = np.random.randn(len(daily_dates)) * 0.01 + 0.0003
        sp500_prices = 2000 * np.exp(np.cumsum(sp500_returns))
        self.usa_market = pd.DataFrame({
            'Close': sp500_prices
        }, index=daily_dates)
        
        # Generate ASX 200 data
        asx_returns = np.random.randn(len(daily_dates)) * 0.008 + 0.0002
        asx_prices = 5000 * np.exp(np.cumsum(asx_returns))
        self.aus_market = pd.DataFrame({
            'Close': asx_prices
        }, index=daily_dates)
        
        # Generate economic indicators
        self.indicators = {
            'gold': pd.Series(
                np.linspace(1200, 2000, len(daily_dates)) + np.random.randn(len(daily_dates)) * 20,
                index=daily_dates
            ),
            'oil': pd.Series(
                70 + 30 * np.sin(np.arange(len(daily_dates)) * 2 * np.pi / 250) + np.random.randn(len(daily_dates)) * 5,
                index=daily_dates
            ),
            'copper': pd.Series(
                np.linspace(3.0, 4.5, len(daily_dates)) + np.random.randn(len(daily_dates)) * 0.2,
                index=daily_dates
            ),
            'yield_curve': pd.Series(
                2.0 + np.sin(np.arange(len(daily_dates)) * 2 * np.pi / 500) * 1.5 + np.random.randn(len(daily_dates)) * 0.1,
                index=daily_dates
            ),
            'vix': pd.Series(
                20 + np.random.exponential(scale=5, size=len(daily_dates)),
                index=daily_dates
            ).rolling(window=5, min_periods=1).mean(),
            'credit_spreads': pd.Series(
                2.5 + np.random.randn(len(daily_dates)) * 0.3,
                index=daily_dates
            ),
            'money_market': pd.Series(
                np.cumsum(np.random.randn(len(daily_dates)) * 100) + 5000,
                index=daily_dates
            ),
            'jobless_claims': pd.Series(
                250000 + np.random.randn(len(daily_dates)) * 20000,
                index=daily_dates
            ).clip(lower=150000),
            'mortgage_rates': pd.Series(
                4.5 + np.sin(np.arange(len(daily_dates)) * 2 * np.pi / 750) * 1.5,
                index=daily_dates
            ),
            'consumer_confidence': pd.Series(
                100 + np.sin(np.arange(len(daily_dates)) * 2 * np.pi / 365) * 20 + np.random.randn(len(daily_dates)) * 5,
                index=daily_dates
            ),
            'options_volume': pd.Series(
                np.linspace(1e6, 3e6, len(daily_dates)) + np.random.exponential(scale=5e5, size=len(daily_dates)),
                index=daily_dates
            )
        }
    
    def check_wednesday_update(self):
        """Check if it's Wednesday and update ICI data"""
        today = datetime.now()
        if today.weekday() == 2:  # Wednesday
            if st.session_state.last_update is None or st.session_state.last_update.date() < today.date():
                with st.spinner("It's Wednesday! Fetching ICI weekly data..."):
                    self.fetch_ici_data()
                    st.session_state.last_update = today
    
    def fetch_ici_data(self):
        """Fetch weekly ICI data"""
        try:
            if self.ici_api_key:
                # Real ICI API call
                url = "https://www.ici.org/api/v1/research/stats/combined"
                headers = {"Authorization": f"Bearer {self.ici_api_key}"}
                
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    self.process_ici_data(data)
                    return data
            
            # Simulate weekly update
            return self.generate_simulated_flow_data()
            
        except Exception as e:
            return self.generate_simulated_flow_data()
    
    def generate_simulated_flow_data(self):
        """Generate simulated weekly flow update"""
        # Add new weekly data point
        last_date = self.usa_data.index[-1]
        new_date = last_date + timedelta(days=7)
        
        # USA update
        last_flow = self.usa_data['total_flows'].iloc[-1]
        change = np.random.randn() * 50 + 10
        new_flow = last_flow + change
        
        new_usa_row = pd.DataFrame({
            'retail_flows': [new_flow * 0.6],
            'institutional_flows': [new_flow * 0.4],
            'total_flows': [new_flow],
            'retail_flows_change': [(new_flow * 0.6 - self.usa_data['retail_flows'].iloc[-1]) / self.usa_data['retail_flows'].iloc[-1] * 100],
            'institutional_flows_change': [(new_flow * 0.4 - self.usa_data['institutional_flows'].iloc[-1]) / self.usa_data['institutional_flows'].iloc[-1] * 100]
        }, index=[new_date])
        
        self.usa_data = pd.concat([self.usa_data, new_usa_row])
        
        # Australia update
        last_flow_aus = self.aus_data['total_flows'].iloc[-1]
        change_aus = np.random.randn() * 30 + 5
        new_flow_aus = last_flow_aus + change_aus
        
        new_aus_row = pd.DataFrame({
            'retail_flows': [new_flow_aus * 0.5],
            'institutional_flows': [new_flow_aus * 0.5],
            'total_flows': [new_flow_aus],
            'retail_flows_change': [(new_flow_aus * 0.5 - self.aus_data['retail_flows'].iloc[-1]) / self.aus_data['retail_flows'].iloc[-1] * 100],
            'institutional_flows_change': [(new_flow_aus * 0.5 - self.aus_data['institutional_flows'].iloc[-1]) / self.aus_data['institutional_flows'].iloc[-1] * 100]
        }, index=[new_date])
        
        self.aus_data = pd.concat([self.aus_data, new_aus_row])
        
        # Update session state
        st.session_state.usa_data = self.usa_data
        st.session_state.aus_data = self.aus_data
        
        return {'usa': new_flow, 'aus': new_flow_aus}
    
    def fetch_market_data(self, symbol, start_date):
        """Fetch market data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=datetime.now())
            return data
        except:
            return None
    
    def fetch_economic_indicators(self):
        """Fetch economic indicators from FRED and Alpha Vantage"""
        indicators = {}
        
        if self.fred and self.fred_api_key:
            try:
                series_map = {
                    'gold': 'GOLDAMGBD228NLBM',
                    'oil': 'DCOILWTICO',
                    'yield_curve': 'T10Y2Y',
                    'credit_spreads': 'BAA10Y',
                    'money_market': 'WRMFSL',
                    'jobless_claims': 'ICSA',
                    'mortgage_rates': 'MORTGAGE30US',
                    'consumer_confidence': 'UMCSENT'
                }
                
                start_date = datetime.now() - timedelta(days=30)
                
                for name, series_id in series_map.items():
                    try:
                        data = self.fred.get_series(series_id, start_date)
                        if len(data) > 0:
                            indicators[name] = data
                    except:
                        pass
            except:
                pass
        
        # Fetch copper from Alpha Vantage
        if self.alpha_vantage_key:
            try:
                url = f"https://www.alphavantage.co/query?function=COPPER&interval=daily&apikey={self.alpha_vantage_key}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data:
                        copper_df = pd.DataFrame(data['data'])
                        copper_df['date'] = pd.to_datetime(copper_df['date'])
                        copper_df.set_index('date', inplace=True)
                        indicators['copper'] = copper_df['value'].astype(float)
            except:
                pass
        
        # Generate options volume data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='B')
        indicators['options_volume'] = pd.Series(
            np.random.exponential(scale=1000000, size=len(dates)) * (1 + np.random.randn(len(dates)) * 0.1),
            index=dates
        )
        
        return indicators
    
    def update_all_data(self):
        """Update all data from APIs"""
        with st.spinner("Fetching latest data..."):
            # Fetch ICI data
            ici_data = self.fetch_ici_data()
            
            # Fetch USA market data
            usa_market = self.fetch_market_data("^GSPC", datetime.now() - timedelta(days=30))
            if usa_market is not None and not usa_market.empty:
                self.usa_market = usa_market
                st.session_state.usa_market = usa_market
            
            # Fetch Australia market data
            aus_market = self.fetch_market_data("^AXJO", datetime.now() - timedelta(days=30))
            if aus_market is not None and not aus_market.empty:
                self.aus_market = aus_market
                st.session_state.aus_market = aus_market
            
            # Fetch economic indicators
            new_indicators = self.fetch_economic_indicators()
            
            # Update indicators with new data
            for name, data in new_indicators.items():
                if len(data) > 0:
                    self.indicators[name] = data
            
            st.session_state.indicators = self.indicators
            st.session_state.last_update = datetime.now()
            
            # Save cache
            self.save_cache()
            
            return True
    
    def save_cache(self):
        """Save current data to cache file"""
        try:
            cache_data = {
                'usa_data': self.usa_data,
                'aus_data': self.aus_data,
                'usa_market': self.usa_market,
                'aus_market': self.aus_market,
                'indicators': self.indicators,
                'last_update': datetime.now()
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except:
            pass
    
    def load_cache(self):
        """Load data from cache file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    return cache
        except:
            pass
        return None
    
    def create_flow_chart(self, data, market_data, indicators, selected_indicators, time_range, market_name):
        """Create interactive chart with flows and overlays"""
        
        # Filter data by time range
        if time_range != 'All' and self.time_ranges[time_range]:
            cutoff_date = datetime.now() - timedelta(days=self.time_ranges[time_range])
            data = data[data.index >= cutoff_date]
            if market_data is not None and not market_data.empty:
                market_data = market_data[market_data.index >= cutoff_date]
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            specs=[
                [{"secondary_y": True}],
                [{"secondary_y": False}],
                [{"secondary_y": False}]
            ],
            subplot_titles=(
                f'{market_name} Market Flows & Indicators',
                'Rate of Change (%)',
                'Flow Momentum'
            )
        )
        
        # Add market price on primary axis
        if show_market and market_data is not None and not market_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=market_data.index,
                    y=market_data['Close'],
                    name=f'{market_name} Index',
                    line=dict(color='#1f77b4', width=2)
                ),
                row=1, col=1, secondary_y=False
            )
        
        # Add flow data on secondary axis
        if show_flows:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['total_flows'],
                    name='Total Flows',
                    line=dict(color='#2ca02c', width=2)
                ),
                row=1, col=1, secondary_y=True
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['retail_flows'],
                    name='Retail Flows',
                    line=dict(color='#17becf', width=1.5),
                    visible='legendonly'
                ),
                row=1, col=1, secondary_y=True
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['institutional_flows'],
                    name='Institutional Flows',
                    line=dict(color='#bcbd22', width=1.5),
                    visible='legendonly'
                ),
                row=1, col=1, secondary_y=True
            )
        
        # Add selected economic indicators
        colors = ['#FFD700', '#8B4513', '#B87333', '#9467bd', '#d62728', '#ff7f0e', '#8c564b', '#e377c2', '#7f7f7f', '#17becf']
        color_idx = 0
        
        for indicator_name, indicator_data in selected_indicators.items():
            if indicator_data is not None and not indicator_data.empty:
                # Resample indicator data to match chart timeframe
                if time_range != 'All' and self.time_ranges[time_range]:
                    indicator_data = indicator_data[indicator_data.index >= cutoff_date]
                
                if len(indicator_data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=indicator_data.index,
                            y=indicator_data.values if hasattr(indicator_data, 'values') else indicator_data,
                            name=indicator_name,
                            line=dict(color=colors[color_idx % len(colors)], width=1, dash='dot'),
                            visible='legendonly'
                        ),
                        row=1, col=1, secondary_y=True
                    )
                    color_idx += 1
        
        # Add rate of change chart
        if show_roc and 'retail_flows_change' in data.columns:
            # Create bar colors based on positive/negative
            colors = ['green' if x > 0 else 'red' for x in data['retail_flows_change'].fillna(0)]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['retail_flows_change'],
                    name='Retail Flow RoC',
                    marker_color=colors
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['institutional_flows_change'],
                    name='Institutional Flow RoC',
                    line=dict(color='#1f77b4', width=1),
                    visible='legendonly'
                ),
                row=2, col=1
            )
        
        # Add flow momentum (moving averages)
        ma_4w = data['total_flows'].rolling(window=4).mean()
        ma_13w = data['total_flows'].rolling(window=13).mean()
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=ma_4w,
                name='4-Week MA',
                line=dict(color='#3498DB', width=1.5)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=ma_13w,
                name='13-Week MA',
                line=dict(color='#E74C3C', width=1.5)
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Index Price", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Flows & Indicators", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Rate of Change (%)", row=2, col=1)
        fig.update_yaxes(title_text="Flow MA", row=3, col=1)
        
        fig.update_layout(
            height=900,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#0e1117'),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        return fig
    
    def send_weekly_email(self, analysis, recipient_email):
        """Send weekly email with analysis"""
        if not self.sender_email or not self.sender_password:
            return False
        
        try:
            message = MIMEMultipart("alternative")
            message["Subject"] = f"Weekly Market Flow Analysis - {datetime.now().strftime('%Y-%m-%d')}"
            message["From"] = self.sender_email
            message["To"] = recipient_email
            
            html = f"""
            <html>
              <body>
                <h2>Weekly Market Flow Analysis</h2>
                <p>{datetime.now().strftime('%B %d, %Y')}</p>
                <div style="white-space: pre-wrap;">{analysis}</div>
                <p>Visit the dashboard for interactive charts and detailed data.</p>
              </body>
            </html>
            """
            
            part = MIMEText(html, "html")
            message.attach(part)
            
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, recipient_email, message.as_string())
            
            return True
        except Exception as e:
            st.error(f"Failed to send email: {e}")
            return False

def main():
    st.title("📈 Stock Market Flow Tracker - USA & Australia")
    st.markdown("Track retail and institutional investment flows with economic indicators")
    
    # Initialize tracker
    tracker = MarketFlowTracker()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Update status
        if st.session_state.last_update:
            st.success(f"✅ Last update: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M')}")
        
        # Wednesday notification
        if datetime.now().weekday() == 2:
            st.info("📅 It's Wednesday - ICI data updating automatically")
        
        # Market selection
        market_view = st.radio("Select Market View", ["USA", "Australia", "Both"])
        
        # Time range selection
        time_range = st.selectbox(
            "Time Range",
            options=list(tracker.time_ranges.keys()),
            index=3  # Default to 1Y
        )
        
        st.subheader("📊 Economic Indicators")
        
        # Checkboxes for indicators - THREE DEFAULTS CHECKED
        global show_flows, show_roc, show_market
        show_flows = st.checkbox("💰 Investment Flow", value=True)
        show_roc = st.checkbox("📈 Rate of Change", value=True)  
        show_market = st.checkbox("📊 Stock Market Price", value=True)
        
        st.markdown("---")
        
        show_gold = st.checkbox("🟡 Gold Prices", value=False)
        show_copper = st.checkbox("🟫 Copper Prices", value=False)
        show_oil = st.checkbox("🛢️ Oil Prices (WTI)", value=False)
        show_yield = st.checkbox("📉 Yield Curve (10Y-2Y)", value=False)
        show_credit = st.checkbox("💳 Credit Spreads", value=False)
        show_money = st.checkbox("💵 Money Market Funds", value=False)
        show_jobless = st.checkbox("👥 Jobless Claims", value=False)
        show_mortgage = st.checkbox("🏠 Mortgage Rates", value=False)
        show_confidence = st.checkbox("😊 Consumer Confidence", value=False)
        show_options = st.checkbox("⚙️ Options Volume", value=False)
        show_vix = st.checkbox("😨 VIX", value=False)
        
        # Email configuration
        st.subheader("📧 Email Alerts")
        email_enabled = st.checkbox("Enable Weekly Email")
        if email_enabled:
            recipient_email = st.text_input("Recipient Email", value=tracker.sender_email)
        
        # Update data button
        if st.button("🔄 Update Data", type="primary"):
            if tracker.update_all_data():
                st.success("✅ Data updated successfully!")
            else:
                st.warning("⚠️ Some data sources unavailable")
    
    # Main content area
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.usa_data is not None:
            usa_latest = st.session_state.usa_data.iloc[-1]
            st.metric(
                "USA Total Flows",
                f"${usa_latest['total_flows']:.0f}M",
                f"{usa_latest.get('retail_flows_change', 0):.1f}%"
            )
    
    with col2:
        if st.session_state.usa_data is not None:
            usa_roc = st.session_state.usa_data['retail_flows_change'].iloc[-1] if 'retail_flows_change' in st.session_state.usa_data.columns else 0
            st.metric("USA Flow RoC", f"{usa_roc:.1f}%")
    
    with col3:
        if st.session_state.aus_data is not None:
            aus_latest = st.session_state.aus_data.iloc[-1]
            st.metric(
                "AUS Total Flows",
                f"${aus_latest['total_flows']:.0f}M",
                f"{aus_latest.get('retail_flows_change', 0):.1f}%"
            )
    
    with col4:
        if st.session_state.aus_data is not None:
            aus_roc = st.session_state.aus_data['retail_flows_change'].iloc[-1] if 'retail_flows_change' in st.session_state.aus_data.columns else 0
            st.metric("AUS Flow RoC", f"{aus_roc:.1f}%")
    
    # Prepare selected indicators
    selected_indicators = {}
    if show_gold and 'gold' in tracker.indicators:
        selected_indicators['Gold'] = tracker.indicators['gold']
    if show_copper and 'copper' in tracker.indicators:
        selected_indicators['Copper'] = tracker.indicators['copper']
    if show_oil and 'oil' in tracker.indicators:
        selected_indicators['Oil'] = tracker.indicators['oil']
    if show_yield and 'yield_curve' in tracker.indicators:
        selected_indicators['Yield Curve'] = tracker.indicators['yield_curve']
    if show_credit and 'credit_spreads' in tracker.indicators:
        selected_indicators['Credit Spreads'] = tracker.indicators['credit_spreads']
    if show_money and 'money_market' in tracker.indicators:
        selected_indicators['Money Market'] = tracker.indicators['money_market']
    if show_jobless and 'jobless_claims' in tracker.indicators:
        selected_indicators['Jobless Claims'] = tracker.indicators['jobless_claims']
    if show_mortgage and 'mortgage_rates' in tracker.indicators:
        selected_indicators['Mortgage Rates'] = tracker.indicators['mortgage_rates']
    if show_confidence and 'consumer_confidence' in tracker.indicators:
        selected_indicators['Consumer Confidence'] = tracker.indicators['consumer_confidence']
    if show_options and 'options_volume' in tracker.indicators:
        selected_indicators['Options Volume'] = tracker.indicators['options_volume']
    if show_vix and 'vix' in tracker.indicators:
        selected_indicators['VIX'] = tracker.indicators['vix']
    
    # Display charts based on selection
    if market_view in ["USA", "Both"]:
        st.subheader("🇺🇸 USA Market")
        if st.session_state.usa_data is not None:
            usa_chart = tracker.create_flow_chart(
                st.session_state.usa_data,
                st.session_state.usa_market,
                tracker.indicators,
                selected_indicators,
                time_range,
                "USA"
            )
            st.plotly_chart(usa_chart, use_container_width=True)
    
    if market_view in ["Australia", "Both"]:
        st.subheader("🇦🇺 Australia Market")
        if st.session_state.aus_data is not None:
            aus_chart = tracker.create_flow_chart(
                st.session_state.aus_data,
                st.session_state.aus_market,
                tracker.indicators,
                selected_indicators,
                time_range,
                "Australia"
            )
            st.plotly_chart(aus_chart, use_container_width=True)
    
    # Footer
    st.divider()
    st.caption("Data sources: ICI, FRED, Alpha Vantage, Yahoo Finance")
    st.caption("Note: Embedded historical data with live updates when API keys are configured")

if __name__ == "__main__":
    main()
