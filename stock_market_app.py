"""
Stock Market Flow Tracker - USA & Australia
Real-time tracking of retail and institutional investment flows with economic indicators
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
from anthropic import Anthropic
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import schedule
import threading
import time
import pickle
from fredapi import Fred

# Page configuration
st.set_page_config(
    page_title="Stock Market Flow Tracker - USA & Australia",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'usa_data' not in st.session_state:
    st.session_state.usa_data = None
if 'aus_data' not in st.session_state:
    st.session_state.aus_data = None
if 'weekly_analyses' not in st.session_state:
    st.session_state.weekly_analyses = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

class MarketFlowTracker:
    def __init__(self):
        # API Keys (store these in Streamlit secrets)
        self.fred_api_key = st.secrets.get("FRED_API_KEY", "")
        self.alpha_vantage_key = st.secrets.get("ALPHA_VANTAGE_KEY", "")
        self.anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        self.ici_api_key = st.secrets.get("ICI_API_KEY", "")
        
        # Initialize FRED
        if self.fred_api_key:
            self.fred = Fred(api_key=self.fred_api_key)
        
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
        
        # Initialize data cache
        self.cache_file = 'market_data_cache.pkl'
        self.load_cache()
    
    def load_cache(self):
        """Load cached data if available"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
            else:
                self.cache = {}
        except:
            self.cache = {}
    
    def save_cache(self):
        """Save data to cache"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except:
            pass
    
    def fetch_ici_data(self):
        """Fetch weekly ICI (Investment Company Institute) data for mutual fund flows"""
        try:
            # Note: ICI data requires subscription. This is a template for the API call
            url = "https://www.ici.org/api/v1/research/stats/combined"
            headers = {"Authorization": f"Bearer {self.ici_api_key}"}
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                # Process ICI data for retail and institutional flows
                return self.process_ici_data(data)
            else:
                # Fallback to simulated data for demonstration
                return self.generate_simulated_flow_data()
        except:
            return self.generate_simulated_flow_data()
    
    def generate_simulated_flow_data(self):
        """Generate simulated flow data for demonstration"""
        dates = pd.date_range(end=datetime.now(), periods=15*52, freq='W')
        
        # Simulate retail and institutional flows with realistic patterns
        np.random.seed(42)
        retail_flows = np.cumsum(np.random.randn(len(dates)) * 5 + 2)
        institutional_flows = np.cumsum(np.random.randn(len(dates)) * 10 + 3)
        
        # Add trend and seasonality
        trend = np.linspace(0, 100, len(dates))
        seasonal = 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 52)
        
        retail_flows = retail_flows + trend + seasonal
        institutional_flows = institutional_flows + trend * 1.5 + seasonal * 0.5
        
        df = pd.DataFrame({
            'date': dates,
            'retail_flows': retail_flows,
            'institutional_flows': institutional_flows,
            'retail_flows_change': np.gradient(retail_flows),
            'institutional_flows_change': np.gradient(institutional_flows)
        })
        
        return df
    
    def fetch_market_data(self, symbol, start_date):
        """Fetch stock market data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=datetime.now())
            return data
        except Exception as e:
            st.error(f"Error fetching market data for {symbol}: {e}")
            return None
    
    def fetch_economic_indicators(self, start_date):
        """Fetch all economic indicators"""
        indicators = {}
        
        try:
            # Gold Prices (GOLDAMGBD228NLBM)
            indicators['gold'] = self.fred.get_series('GOLDAMGBD228NLBM', start_date)
            
            # WTI Oil Prices (DCOILWTICO)
            indicators['oil'] = self.fred.get_series('DCOILWTICO', start_date)
            
            # Yield Curve 10Y-2Y (T10Y2Y)
            indicators['yield_curve'] = self.fred.get_series('T10Y2Y', start_date)
            
            # Credit Spreads (BAA10Y)
            indicators['credit_spreads'] = self.fred.get_series('BAA10Y', start_date)
            
            # Money Market Funds (WRMFSL)
            indicators['money_market'] = self.fred.get_series('WRMFSL', start_date)
            
            # Initial Jobless Claims (ICSA)
            indicators['jobless_claims'] = self.fred.get_series('ICSA', start_date)
            
            # Mortgage Rates (MORTGAGE30US)
            indicators['mortgage_rates'] = self.fred.get_series('MORTGAGE30US', start_date)
            
            # Consumer Confidence (UMCSENT)
            indicators['consumer_confidence'] = self.fred.get_series('UMCSENT', start_date)
            
        except Exception as e:
            st.warning(f"Error fetching some indicators: {e}")
        
        # Fetch Copper prices from Alpha Vantage
        try:
            url = f"https://www.alphavantage.co/query?function=COPPER&interval=daily&apikey={self.alpha_vantage_key}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                # Process copper data
                copper_df = pd.DataFrame(data['data'])
                copper_df['date'] = pd.to_datetime(copper_df['date'])
                copper_df.set_index('date', inplace=True)
                indicators['copper'] = copper_df['value'].astype(float)
        except:
            pass
        
        # Fetch options trading volume (simulated for now)
        indicators['options_volume'] = self.generate_options_volume_data(start_date)
        
        return indicators
    
    def generate_options_volume_data(self, start_date):
        """Generate simulated options volume data"""
        dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')
        volume = np.random.exponential(scale=1000000, size=len(dates)) * (1 + np.random.randn(len(dates)) * 0.1)
        return pd.Series(volume, index=dates)
    
    def calculate_rate_of_change(self, data, window=4):
        """Calculate rate of change for flows"""
        return data.pct_change(periods=window) * 100
    
    def generate_ai_analysis(self, usa_data, aus_data, indicators):
        """Generate weekly AI analysis using Claude"""
        if not self.anthropic_key:
            return "API key not configured for AI analysis"
        
        try:
            client = Anthropic(api_key=self.anthropic_key)
            
            # Prepare data summary for analysis
            summary = f"""
            Weekly Market Analysis - {datetime.now().strftime('%Y-%m-%d')}
            
            USA Market:
            - Retail Flow Change: {usa_data['retail_flows_change'].iloc[-1]:.2f}
            - Institutional Flow Change: {usa_data['institutional_flows_change'].iloc[-1]:.2f}
            - Market Performance: {usa_data.get('market_return', 0):.2f}%
            
            Australia Market:
            - Retail Flow Change: {aus_data['retail_flows_change'].iloc[-1]:.2f}
            - Institutional Flow Change: {aus_data['institutional_flows_change'].iloc[-1]:.2f}
            - Market Performance: {aus_data.get('market_return', 0):.2f}%
            
            Key Indicators:
            - Gold: ${indicators.get('gold', pd.Series()).iloc[-1] if not indicators.get('gold', pd.Series()).empty else 'N/A'}
            - Oil: ${indicators.get('oil', pd.Series()).iloc[-1] if not indicators.get('oil', pd.Series()).empty else 'N/A'}
            - Yield Curve: {indicators.get('yield_curve', pd.Series()).iloc[-1] if not indicators.get('yield_curve', pd.Series()).empty else 'N/A'}
            
            Please provide a concise weekly analysis focusing on:
            1. Key flow trends and their implications
            2. Notable rate of change signals
            3. Economic indicator correlations
            4. Market outlook for the week ahead
            """
            
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=500,
                messages=[
                    {"role": "user", "content": summary}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"AI analysis unavailable: {str(e)}"
    
    def create_flow_chart(self, data, market_data, indicators, selected_indicators, time_range, market_name):
        """Create interactive chart with flows and overlays"""
        
        # Filter data by time range
        if time_range != 'All' and self.time_ranges[time_range]:
            cutoff_date = datetime.now() - timedelta(days=self.time_ranges[time_range])
            data = data[data['date'] >= cutoff_date]
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
            subplot_titles=(f'{market_name} Market Flows & Indicators', 'Rate of Change')
        )
        
        # Add market price on primary axis
        if market_data is not None and not market_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=market_data.index,
                    y=market_data['Close'],
                    name=f'{market_name} Index',
                    line=dict(color='black', width=2),
                    yaxis='y'
                ),
                row=1, col=1, secondary_y=False
            )
        
        # Add flow data on secondary axis
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['retail_flows'],
                name='Retail Flows',
                line=dict(color='blue', width=1.5),
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )
        
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['institutional_flows'],
                name='Institutional Flows',
                line=dict(color='green', width=1.5),
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Add selected economic indicators
        colors = ['gold', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, (key, name) in enumerate(selected_indicators.items()):
            if name and key in indicators and not indicators[key].empty:
                fig.add_trace(
                    go.Scatter(
                        x=indicators[key].index,
                        y=indicators[key].values,
                        name=name,
                        line=dict(color=colors[i % len(colors)], width=1, dash='dot'),
                        visible='legendonly',  # Hidden by default
                        yaxis='y2'
                    ),
                    row=1, col=1, secondary_y=True
                )
        
        # Add rate of change chart
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['retail_flows_change'],
                name='Retail Flow RoC',
                line=dict(color='lightblue', width=1),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['institutional_flows_change'],
                name='Institutional Flow RoC',
                line=dict(color='lightgreen', width=1),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        # Add rate change markers
        pos_changes = data[data['retail_flows_change'] > data['retail_flows_change'].quantile(0.9)]
        neg_changes = data[data['retail_flows_change'] < data['retail_flows_change'].quantile(0.1)]
        
        fig.add_trace(
            go.Scatter(
                x=pos_changes['date'],
                y=pos_changes['retail_flows_change'],
                mode='markers',
                marker=dict(color='green', size=8, symbol='triangle-up'),
                name='Positive Rate Change',
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=neg_changes['date'],
                y=neg_changes['retail_flows_change'],
                mode='markers',
                marker=dict(color='red', size=8, symbol='triangle-down'),
                name='Negative Rate Change',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Index Price", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Flows & Indicators", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Rate of Change (%)", row=2, col=1)
        
        fig.update_layout(
            height=800,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def send_weekly_email(self, analysis, recipient_email):
        """Send weekly email with analysis"""
        try:
            sender_email = st.secrets.get("SENDER_EMAIL", "")
            sender_password = st.secrets.get("SENDER_PASSWORD", "")
            
            message = MIMEMultipart("alternative")
            message["Subject"] = f"Weekly Market Flow Analysis - {datetime.now().strftime('%Y-%m-%d')}"
            message["From"] = sender_email
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
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient_email, message.as_string())
            
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
        
        # Market selection
        market_view = st.radio("Select Market View", ["USA", "Australia", "Both"])
        
        # Time range selection
        time_range = st.selectbox(
            "Time Range",
            options=list(tracker.time_ranges.keys()),
            index=3  # Default to 1Y
        )
        
        st.subheader("📊 Economic Indicators")
        
        # Checkboxes for indicators
        show_gold = st.checkbox("Gold Prices", value=True)
        show_copper = st.checkbox("Copper Prices", value=True)
        show_oil = st.checkbox("Oil Prices (WTI)", value=True)
        show_yield = st.checkbox("Yield Curve (10Y-2Y)", value=True)
        show_credit = st.checkbox("Credit Spreads", value=False)
        show_money = st.checkbox("Money Market Funds", value=False)
        show_jobless = st.checkbox("Jobless Claims", value=False)
        show_mortgage = st.checkbox("Mortgage Rates", value=False)
        show_confidence = st.checkbox("Consumer Confidence", value=False)
        show_options = st.checkbox("Options Volume", value=True)
        
        # Y-axis display options
        st.subheader("📏 Y-Axis Display")
        show_dollars = st.checkbox("Show in Dollars ($)", value=True)
        show_percentage = st.checkbox("Show Percentage (%)", value=False)
        
        # Email configuration
        st.subheader("📧 Email Alerts")
        email_enabled = st.checkbox("Enable Weekly Email")
        if email_enabled:
            recipient_email = st.text_input("Recipient Email")
        
        # Update data button
        if st.button("🔄 Update Data", type="primary"):
            with st.spinner("Fetching latest data..."):
                # Fetch all data
                start_date = datetime.now() - timedelta(days=15*365)
                
                # USA Data
                usa_flows = tracker.fetch_ici_data()
                usa_market = tracker.fetch_market_data("^GSPC", start_date)  # S&P 500
                
                # Australia Data
                aus_flows = tracker.generate_simulated_flow_data()  # Replace with actual ASX data
                aus_market = tracker.fetch_market_data("^AXJO", start_date)  # ASX 200
                
                # Economic indicators
                indicators = tracker.fetch_economic_indicators(start_date)
                
                # Store in session state
                st.session_state.usa_data = usa_flows
                st.session_state.aus_data = aus_flows
                st.session_state.usa_market = usa_market
                st.session_state.aus_market = aus_market
                st.session_state.indicators = indicators
                
                # Generate AI analysis
                analysis = tracker.generate_ai_analysis(usa_flows, aus_flows, indicators)
                st.session_state.current_analysis = analysis
                st.session_state.weekly_analyses.append({
                    'date': datetime.now(),
                    'analysis': analysis
                })
                
                st.success("Data updated successfully!")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # AI Analysis Section
        st.header("🤖 Claude AI Weekly Analysis")
        
        if st.session_state.current_analysis:
            analysis_date = datetime.now().strftime('%B %d, %Y')
            st.subheader(f"Analysis for {analysis_date}")
            st.info(st.session_state.current_analysis)
            
            # Show previous analyses
            if len(st.session_state.weekly_analyses) > 1:
                with st.expander("📅 View Previous Analyses"):
                    for analysis in reversed(st.session_state.weekly_analyses[:-1]):
                        st.write(f"**{analysis['date'].strftime('%B %d, %Y')}**")
                        st.write(analysis['analysis'])
                        st.divider()
    
    with col2:
        # Quick stats
        st.header("📊 Quick Stats")
        if st.session_state.usa_data is not None:
            usa_latest = st.session_state.usa_data.iloc[-1]
            st.metric("USA Retail Flow Change", f"{usa_latest['retail_flows_change']:.2f}%")
            st.metric("USA Institutional Flow Change", f"{usa_latest['institutional_flows_change']:.2f}%")
        
        if st.session_state.aus_data is not None:
            aus_latest = st.session_state.aus_data.iloc[-1]
            st.metric("AUS Retail Flow Change", f"{aus_latest['retail_flows_change']:.2f}%")
            st.metric("AUS Institutional Flow Change", f"{aus_latest['institutional_flows_change']:.2f}%")
    
    # Charts section
    st.header("📈 Interactive Charts")
    
    # Prepare selected indicators
    selected_indicators = {}
    if show_gold:
        selected_indicators['gold'] = 'Gold Prices'
    if show_copper:
        selected_indicators['copper'] = 'Copper Prices'
    if show_oil:
        selected_indicators['oil'] = 'Oil Prices'
    if show_yield:
        selected_indicators['yield_curve'] = 'Yield Curve'
    if show_credit:
        selected_indicators['credit_spreads'] = 'Credit Spreads'
    if show_money:
        selected_indicators['money_market'] = 'Money Market Funds'
    if show_jobless:
        selected_indicators['jobless_claims'] = 'Jobless Claims'
    if show_mortgage:
        selected_indicators['mortgage_rates'] = 'Mortgage Rates'
    if show_confidence:
        selected_indicators['consumer_confidence'] = 'Consumer Confidence'
    if show_options:
        selected_indicators['options_volume'] = 'Options Volume'
    
    # Display charts based on selection
    if market_view in ["USA", "Both"]:
        st.subheader("🇺🇸 USA Market")
        if st.session_state.usa_data is not None and st.session_state.indicators:
            usa_chart = tracker.create_flow_chart(
                st.session_state.usa_data,
                st.session_state.usa_market if 'usa_market' in st.session_state else None,
                st.session_state.indicators,
                selected_indicators,
                time_range,
                "USA"
            )
            st.plotly_chart(usa_chart, use_container_width=True)
        else:
            st.info("Click 'Update Data' to load USA market data")
    
    if market_view in ["Australia", "Both"]:
        st.subheader("🇦🇺 Australia Market")
        if st.session_state.aus_data is not None and st.session_state.indicators:
            aus_chart = tracker.create_flow_chart(
                st.session_state.aus_data,
                st.session_state.aus_market if 'aus_market' in st.session_state else None,
                st.session_state.indicators,
                selected_indicators,
                time_range,
                "Australia"
            )
            st.plotly_chart(aus_chart, use_container_width=True)
        else:
            st.info("Click 'Update Data' to load Australia market data")
    
    # Footer
    st.divider()
    st.caption("Data sources: ICI (Investment Company Institute), FRED, Alpha Vantage, Yahoo Finance")
    st.caption("Note: Some data is simulated for demonstration purposes. Configure API keys in Streamlit secrets for live data.")

if __name__ == "__main__":
    main()
