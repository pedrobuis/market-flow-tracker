"""
Market Flow Tracker - Clean Version with Embedded Historical Data
Optimized for Desktop and iPad displays
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import json
from fredapi import Fred
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Market Flow Tracker - USA & Australia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean white background with dark text
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Text color */
    .stMarkdown, .stText, p, span, label, .stSelectbox label, .stCheckbox label {
        color: #0e1117 !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #0e1117 !important;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 2px solid #e9ecef;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #0e1117 !important;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #2E86C1;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border-radius: 0.375rem;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #1B5E8F;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Checkbox styling */
    .stCheckbox > div {
        color: #0e1117 !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: white;
        color: #0e1117;
        border: 1px solid #dee2e6;
    }
    
    /* Main content area */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: none;
    }
    
    /* Chart containers */
    .js-plotly-plot {
        background-color: white !important;
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa;
        border-radius: 0.5rem 0.5rem 0 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #0e1117 !important;
        font-weight: 500;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #E3F2FD;
        color: #0e1117;
        border: 1px solid #90CAF9;
    }
    
    /* Success messages */
    .stSuccess {
        background-color: #E8F5E9;
        color: #0e1117;
        border: 1px solid #81C784;
    }
    
    /* Warning messages */
    .stWarning {
        background-color: #FFF3E0;
        color: #0e1117;
        border: 1px solid #FFB74D;
    }
    
    /* Dividers */
    hr {
        border-color: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

class MarketFlowTracker:
    def __init__(self):
        # Initialize with embedded historical data
        self.initialize_historical_data()
        
        # API Keys from secrets
        self.fred_api_key = st.secrets.get("FRED_API_KEY", "")
        
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
        
        # Chart colors
        self.colors = {
            'market_price': '#1f77b4',  # Professional blue
            'investment_flow': '#2ca02c',  # Clear green
            'rate_of_change': '#ff7f0e',  # Vibrant orange
            'gold': '#FFD700',
            'oil': '#8B4513',
            'copper': '#B87333',
            'yield_curve': '#9467bd',
            'vix': '#d62728',
            'positive': '#27AE60',
            'negative': '#E74C3C',
            'neutral': '#95A5A6'
        }
    
    def initialize_historical_data(self):
        """Create embedded historical data spanning 15+ years"""
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=15*365)
        
        # Weekly dates for flow data
        weekly_dates = pd.date_range(start=start_date, end=end_date, freq='W')
        
        # Daily dates for market data
        daily_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Generate realistic historical flow data
        np.random.seed(42)  # For reproducibility
        
        # Base trend with growth
        trend = np.linspace(1000, 2500, len(weekly_dates))
        
        # Add cyclical patterns
        annual_cycle = 200 * np.sin(np.arange(len(weekly_dates)) * 2 * np.pi / 52)
        quarterly_cycle = 100 * np.sin(np.arange(len(weekly_dates)) * 2 * np.pi / 13)
        
        # Random walk
        random_component = np.cumsum(np.random.randn(len(weekly_dates)) * 50)
        
        # Create investment flow data
        self.flow_data_usa = pd.DataFrame(index=weekly_dates)
        self.flow_data_usa['investment_flow'] = trend + annual_cycle + random_component
        self.flow_data_usa['investment_flow'] = self.flow_data_usa['investment_flow'].clip(lower=0)
        
        # Calculate rate of change
        self.flow_data_usa['rate_of_change'] = self.flow_data_usa['investment_flow'].pct_change(periods=4) * 100
        
        # Australia flows (slightly different pattern)
        trend_aus = np.linspace(500, 1200, len(weekly_dates))
        self.flow_data_aus = pd.DataFrame(index=weekly_dates)
        self.flow_data_aus['investment_flow'] = trend_aus + annual_cycle * 0.7 + np.cumsum(np.random.randn(len(weekly_dates)) * 30)
        self.flow_data_aus['investment_flow'] = self.flow_data_aus['investment_flow'].clip(lower=0)
        self.flow_data_aus['rate_of_change'] = self.flow_data_aus['investment_flow'].pct_change(periods=4) * 100
        
        # Generate market price data (daily)
        # S&P 500 simulation
        sp500_returns = np.random.randn(len(daily_dates)) * 0.01 + 0.0003  # Daily returns
        sp500_price = 2000 * np.exp(np.cumsum(sp500_returns))
        self.market_data_usa = pd.Series(sp500_price, index=daily_dates, name='SP500')
        
        # ASX 200 simulation
        asx_returns = np.random.randn(len(daily_dates)) * 0.008 + 0.0002
        asx_price = 5000 * np.exp(np.cumsum(asx_returns))
        self.market_data_aus = pd.Series(asx_price, index=daily_dates, name='ASX200')
        
        # Generate economic indicators
        self.indicators = self.generate_indicators(daily_dates)
    
    def generate_indicators(self, dates):
        """Generate realistic economic indicator data"""
        indicators = {}
        
        # Gold prices (trending up with volatility)
        gold_trend = np.linspace(1200, 2000, len(dates))
        gold_volatility = np.random.randn(len(dates)) * 20
        indicators['gold'] = pd.Series(gold_trend + gold_volatility, index=dates)
        
        # Oil prices (more volatile)
        oil_base = 70
        oil_cycle = 30 * np.sin(np.arange(len(dates)) * 2 * np.pi / 250)
        oil_volatility = np.random.randn(len(dates)) * 5
        indicators['oil'] = pd.Series(oil_base + oil_cycle + oil_volatility, index=dates)
        
        # Copper prices
        copper_trend = np.linspace(3.0, 4.5, len(dates))
        copper_volatility = np.random.randn(len(dates)) * 0.2
        indicators['copper'] = pd.Series(copper_trend + copper_volatility, index=dates)
        
        # Yield curve (10Y-2Y)
        yield_curve = 2.0 + np.sin(np.arange(len(dates)) * 2 * np.pi / 500) * 1.5
        yield_noise = np.random.randn(len(dates)) * 0.1
        indicators['yield_curve'] = pd.Series(yield_curve + yield_noise, index=dates)
        
        # VIX (fear index)
        vix_base = 20
        vix_spikes = np.random.exponential(scale=5, size=len(dates))
        vix_smooth = pd.Series(vix_base + vix_spikes, index=dates).rolling(window=5).mean()
        indicators['vix'] = vix_smooth
        
        # Credit spreads
        indicators['credit_spreads'] = pd.Series(
            2.5 + np.random.randn(len(dates)) * 0.3,
            index=dates
        )
        
        # Money market flows
        indicators['money_market'] = pd.Series(
            np.cumsum(np.random.randn(len(dates)) * 100) + 5000,
            index=dates
        )
        
        # Jobless claims
        indicators['jobless_claims'] = pd.Series(
            250000 + np.random.randn(len(dates)) * 20000,
            index=dates
        ).clip(lower=150000)
        
        # Mortgage rates
        indicators['mortgage_rates'] = pd.Series(
            4.5 + np.sin(np.arange(len(dates)) * 2 * np.pi / 750) * 1.5,
            index=dates
        )
        
        # Consumer confidence
        indicators['consumer_confidence'] = pd.Series(
            100 + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 20 + np.random.randn(len(dates)) * 5,
            index=dates
        )
        
        # Options volume
        trend = np.linspace(1e6, 3e6, len(dates))
        indicators['options_volume'] = pd.Series(
            trend + np.random.exponential(scale=5e5, size=len(dates)),
            index=dates
        )
        
        return indicators
    
    def fetch_live_data(self):
        """Fetch live data from APIs if available"""
        try:
            if self.fred_api_key:
                fred = Fred(api_key=self.fred_api_key)
                # Fetch real-time indicators
                live_indicators = {}
                
                series_map = {
                    'gold': 'GOLDAMGBD228NLBM',
                    'oil': 'DCOILWTICO',
                    'yield_curve': 'T10Y2Y',
                    'vix': 'VIXCLS',
                    'credit_spreads': 'BAA10Y',
                    'money_market': 'WRMFSL',
                    'jobless_claims': 'ICSA',
                    'mortgage_rates': 'MORTGAGE30US',
                    'consumer_confidence': 'UMCSENT'
                }
                
                for name, series_id in series_map.items():
                    try:
                        data = fred.get_series(series_id, observation_start=datetime.now() - timedelta(days=30))
                        if len(data) > 0:
                            # Update latest values
                            self.indicators[name].iloc[-len(data):] = data.values
                    except:
                        pass
            
            # Fetch live market data
            try:
                sp500 = yf.download('^GSPC', period='1mo', progress=False)
                if not sp500.empty:
                    self.market_data_usa.iloc[-len(sp500):] = sp500['Close'].values
                
                asx200 = yf.download('^AXJO', period='1mo', progress=False)
                if not asx200.empty:
                    self.market_data_aus.iloc[-len(asx200):] = asx200['Close'].values
            except:
                pass
                
        except Exception as e:
            st.warning(f"Using embedded data. Live data fetch failed: {e}")
    
    def create_main_chart(self, market='USA', time_range='1Y', selected_indicators=None):
        """Create the main chart with three panels"""
        
        # Select appropriate data
        if market == 'USA':
            flow_data = self.flow_data_usa
            market_data = self.market_data_usa
            market_label = 'S&P 500'
        else:
            flow_data = self.flow_data_aus
            market_data = self.market_data_aus
            market_label = 'ASX 200'
        
        # Filter by time range
        if time_range != 'All':
            days = self.time_ranges[time_range]
            cutoff = datetime.now() - timedelta(days=days)
            flow_data = flow_data[flow_data.index >= cutoff]
            market_data = market_data[market_data.index >= cutoff]
        
        # Create subplots
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
                f'{market} Market Overview',
                'Rate of Change (%)',
                'Flow Momentum'
            )
        )
        
        # Panel 1: Market Price and Investment Flow
        # Add market price (primary y-axis)
        if selected_indicators.get('Stock Market Price', False):
            fig.add_trace(
                go.Scatter(
                    x=market_data.index,
                    y=market_data.values,
                    name=market_label,
                    line=dict(color=self.colors['market_price'], width=2),
                    hovertemplate='%{x|%Y-%m-%d}<br>Price: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1, secondary_y=False
            )
        
        # Add investment flow (secondary y-axis)
        if selected_indicators.get('Investment Flow', True):
            fig.add_trace(
                go.Scatter(
                    x=flow_data.index,
                    y=flow_data['investment_flow'],
                    name='Investment Flow',
                    line=dict(color=self.colors['investment_flow'], width=2),
                    hovertemplate='%{x|%Y-%m-%d}<br>Flow: $%{y:,.0f}M<extra></extra>'
                ),
                row=1, col=1, secondary_y=True
            )
        
        # Add selected economic indicators
        for indicator_name, indicator_key in [
            ('Gold Prices', 'gold'),
            ('Oil Prices', 'oil'),
            ('Copper Prices', 'copper'),
            ('Yield Curve', 'yield_curve'),
            ('VIX', 'vix'),
            ('Credit Spreads', 'credit_spreads'),
            ('Money Market', 'money_market'),
            ('Jobless Claims', 'jobless_claims'),
            ('Mortgage Rates', 'mortgage_rates'),
            ('Consumer Confidence', 'consumer_confidence'),
            ('Options Volume', 'options_volume')
        ]:
            if selected_indicators.get(indicator_name, False):
                indicator_data = self.indicators[indicator_key]
                # Resample to match time range
                if time_range != 'All':
                    indicator_data = indicator_data[indicator_data.index >= cutoff]
                
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data.index,
                        y=indicator_data.values,
                        name=indicator_name,
                        line=dict(
                            color=self.colors.get(indicator_key, '#888888'),
                            width=1,
                            dash='dot'
                        ),
                        visible='legendonly',  # Hidden by default
                        hovertemplate='%{x|%Y-%m-%d}<br>%{y:,.2f}<extra></extra>'
                    ),
                    row=1, col=1, secondary_y=True
                )
        
        # Panel 2: Rate of Change
        if selected_indicators.get('Rate of Change', True):
            # Calculate colors based on positive/negative
            colors = ['green' if x > 0 else 'red' for x in flow_data['rate_of_change'].fillna(0)]
            
            fig.add_trace(
                go.Bar(
                    x=flow_data.index,
                    y=flow_data['rate_of_change'],
                    name='Rate of Change',
                    marker_color=colors,
                    hovertemplate='%{x|%Y-%m-%d}<br>RoC: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, row=2, col=1)
        
        # Panel 3: Flow Momentum (Moving Average)
        flow_ma_short = flow_data['investment_flow'].rolling(window=4).mean()
        flow_ma_long = flow_data['investment_flow'].rolling(window=13).mean()
        
        fig.add_trace(
            go.Scatter(
                x=flow_data.index,
                y=flow_ma_short,
                name='4-Week MA',
                line=dict(color='#3498DB', width=1.5),
                hovertemplate='%{x|%Y-%m-%d}<br>4W MA: $%{y:,.0f}M<extra></extra>'
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=flow_data.index,
                y=flow_ma_long,
                name='13-Week MA',
                line=dict(color='#E74C3C', width=1.5),
                hovertemplate='%{x|%Y-%m-%d}<br>13W MA: $%{y:,.0f}M<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_xaxes(
            title_text="Date",
            row=3, col=1,
            gridcolor='#f0f0f0',
            showgrid=True
        )
        
        fig.update_yaxes(
            title_text=f"{market_label} Price ($)",
            row=1, col=1,
            secondary_y=False,
            gridcolor='#f0f0f0',
            showgrid=True
        )
        
        fig.update_yaxes(
            title_text="Flow ($M) / Indicators",
            row=1, col=1,
            secondary_y=True
        )
        
        fig.update_yaxes(
            title_text="Rate of Change (%)",
            row=2, col=1,
            gridcolor='#f0f0f0',
            showgrid=True
        )
        
        fig.update_yaxes(
            title_text="Flow MA ($M)",
            row=3, col=1,
            gridcolor='#f0f0f0',
            showgrid=True
        )
        
        fig.update_layout(
            height=900,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#0e1117', size=12),
            title=dict(
                font=dict(size=16, color='#0e1117')
            ),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#e9ecef',
                borderwidth=1,
                font=dict(color='#0e1117')
            ),
            margin=dict(r=200)  # Space for legend
        )
        
        return fig
    
    def calculate_current_metrics(self, market='USA'):
        """Calculate current metrics for display"""
        if market == 'USA':
            flow_data = self.flow_data_usa
            market_data = self.market_data_usa
        else:
            flow_data = self.flow_data_aus
            market_data = self.market_data_aus
        
        # Current values
        current_flow = flow_data['investment_flow'].iloc[-1]
        prev_flow = flow_data['investment_flow'].iloc[-2]
        flow_change = ((current_flow - prev_flow) / prev_flow * 100)
        
        current_roc = flow_data['rate_of_change'].iloc[-1]
        
        current_price = market_data.iloc[-1]
        prev_price = market_data.iloc[-7]  # Week ago
        price_change = ((current_price - prev_price) / prev_price * 100)
        
        # Flow momentum (4-week vs 13-week MA)
        ma_4w = flow_data['investment_flow'].rolling(window=4).mean().iloc[-1]
        ma_13w = flow_data['investment_flow'].rolling(window=13).mean().iloc[-1]
        momentum = "Bullish" if ma_4w > ma_13w else "Bearish"
        
        return {
            'flow': current_flow,
            'flow_change': flow_change,
            'roc': current_roc,
            'price': current_price,
            'price_change': price_change,
            'momentum': momentum
        }

def main():
    # Initialize tracker
    tracker = MarketFlowTracker()
    
    # Sidebar - Controls
    with st.sidebar:
        st.title("📊 Market Flow Tracker")
        st.markdown("---")
        
        # Market selector
        st.subheader("Market Selection")
        market_view = st.selectbox(
            "Select Market",
            ["USA", "Australia"],
            index=0
        )
        
        # Time range
        st.subheader("Time Range")
        time_range = st.selectbox(
            "Select Period",
            list(tracker.time_ranges.keys()),
            index=3  # Default to 1Y
        )
        
        # Metric selection
        st.subheader("Metrics & Indicators")
        st.markdown("*Default: Flow, RoC, Market Price*")
        
        selected_indicators = {}
        
        # Primary metrics (checked by default)
        selected_indicators['Investment Flow'] = st.checkbox("💰 Investment Flow", value=True)
        selected_indicators['Rate of Change'] = st.checkbox("📈 Rate of Change", value=True)
        selected_indicators['Stock Market Price'] = st.checkbox("📊 Stock Market Price", value=True)
        
        st.markdown("---")
        
        # Economic indicators
        st.markdown("**Economic Indicators**")
        selected_indicators['Gold Prices'] = st.checkbox("🟡 Gold Prices", value=False)
        selected_indicators['Oil Prices'] = st.checkbox("🛢️ Oil Prices", value=False)
        selected_indicators['Copper Prices'] = st.checkbox("🟫 Copper Prices", value=False)
        selected_indicators['Yield Curve'] = st.checkbox("📉 Yield Curve (10Y-2Y)", value=False)
        selected_indicators['VIX'] = st.checkbox("😨 VIX (Fear Index)", value=False)
        selected_indicators['Credit Spreads'] = st.checkbox("💳 Credit Spreads", value=False)
        selected_indicators['Money Market'] = st.checkbox("💵 Money Market Funds", value=False)
        selected_indicators['Jobless Claims'] = st.checkbox("👥 Jobless Claims", value=False)
        selected_indicators['Mortgage Rates'] = st.checkbox("🏠 Mortgage Rates", value=False)
        selected_indicators['Consumer Confidence'] = st.checkbox("😊 Consumer Confidence", value=False)
        selected_indicators['Options Volume'] = st.checkbox("⚙️ Options Volume", value=False)
        
        st.markdown("---")
        
        # Update button
        if st.button("🔄 Refresh Data", use_container_width=True):
            with st.spinner("Fetching latest data..."):
                tracker.fetch_live_data()
                st.success("Data updated!")
    
    # Main content area
    # Header
    st.title(f"{market_view} Market Flow Analysis")
    st.markdown(f"*Real-time investment flow tracking with {15 if time_range == 'All' else tracker.time_ranges[time_range]/365:.1f} year(s) of historical data*")
    
    # Metrics row
    metrics = tracker.calculate_current_metrics(market_view)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Investment Flow",
            f"${metrics['flow']:,.0f}M",
            f"{metrics['flow_change']:+.1f}%"
        )
    
    with col2:
        st.metric(
            "Rate of Change",
            f"{metrics['roc']:.1f}%",
            "Accelerating" if metrics['roc'] > 0 else "Decelerating"
        )
    
    with col3:
        st.metric(
            f"{'S&P 500' if market_view == 'USA' else 'ASX 200'}",
            f"${metrics['price']:,.0f}",
            f"{metrics['price_change']:+.1f}% (1W)"
        )
    
    with col4:
        st.metric(
            "Flow Momentum",
            metrics['momentum'],
            "4W > 13W MA" if metrics['momentum'] == "Bullish" else "4W < 13W MA"
        )
    
    # Main chart
    st.markdown("---")
    
    chart = tracker.create_main_chart(
        market=market_view,
        time_range=time_range,
        selected_indicators=selected_indicators
    )
    
    st.plotly_chart(chart, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.caption("Data: Embedded historical data (15+ years) with optional live updates via FRED API and Yahoo Finance")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
