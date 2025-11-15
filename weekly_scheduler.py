"""
Weekly Scheduler for Market Data Updates
Runs as a background service to fetch ICI data and send email alerts
"""

import schedule
import time
import requests
import json
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import pickle
import os
from anthropic import Anthropic
import yfinance as yf
from fredapi import Fred

class WeeklyMarketUpdater:
    def __init__(self, config_file='config.json'):
        """Initialize with configuration"""
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.fred = Fred(api_key=self.config['FRED_API_KEY'])
        self.anthropic_client = Anthropic(api_key=self.config['ANTHROPIC_API_KEY'])
        self.data_file = 'market_data.pkl'
        self.recipients = self.config.get('email_recipients', [])
        
    def fetch_ici_weekly_data(self):
        """Fetch latest ICI weekly flow data"""
        try:
            # ICI releases data every Wednesday
            url = "https://www.ici.org/api/v1/research/stats/combined"
            headers = {"Authorization": f"Bearer {self.config['ICI_API_KEY']}"}
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return self.process_ici_data(data)
            else:
                print(f"Failed to fetch ICI data: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching ICI data: {e}")
            return None
    
    def process_ici_data(self, raw_data):
        """Process raw ICI data into structured format"""
        processed = {
            'date': datetime.now(),
            'retail_flows': {
                'equity': raw_data.get('equity_retail', 0),
                'bond': raw_data.get('bond_retail', 0),
                'hybrid': raw_data.get('hybrid_retail', 0),
                'total': raw_data.get('total_retail', 0)
            },
            'institutional_flows': {
                'equity': raw_data.get('equity_institutional', 0),
                'bond': raw_data.get('bond_institutional', 0),
                'hybrid': raw_data.get('hybrid_institutional', 0),
                'total': raw_data.get('total_institutional', 0)
            }
        }
        return processed
    
    def fetch_market_indices(self):
        """Fetch latest market index values"""
        indices = {}
        try:
            # USA - S&P 500
            sp500 = yf.Ticker("^GSPC")
            indices['SP500'] = sp500.history(period="1d")['Close'].iloc[-1]
            
            # Australia - ASX 200
            asx200 = yf.Ticker("^AXJO")
            indices['ASX200'] = asx200.history(period="1d")['Close'].iloc[-1]
            
        except Exception as e:
            print(f"Error fetching market indices: {e}")
        
        return indices
    
    def fetch_economic_indicators(self):
        """Fetch latest economic indicator values"""
        indicators = {}
        try:
            # Get latest values for key indicators
            series_ids = {
                'gold': 'GOLDAMGBD228NLBM',
                'oil': 'DCOILWTICO',
                'yield_curve': 'T10Y2Y',
                'credit_spreads': 'BAA10Y',
                'money_market': 'WRMFSL',
                'jobless_claims': 'ICSA',
                'mortgage_rates': 'MORTGAGE30US',
                'consumer_confidence': 'UMCSENT'
            }
            
            for name, series_id in series_ids.items():
                try:
                    series = self.fred.get_series(series_id, limit=1)
                    if not series.empty:
                        indicators[name] = series.iloc[-1]
                except:
                    pass
                    
        except Exception as e:
            print(f"Error fetching indicators: {e}")
        
        return indicators
    
    def calculate_flow_metrics(self, current_data, historical_data):
        """Calculate flow rate of change and other metrics"""
        metrics = {}
        
        if historical_data and len(historical_data) > 0:
            # Get last week's data
            last_week = historical_data[-1] if len(historical_data) > 0 else None
            
            if last_week:
                # Calculate week-over-week change
                retail_change = ((current_data['retail_flows']['total'] - 
                                last_week['retail_flows']['total']) / 
                               last_week['retail_flows']['total'] * 100)
                
                institutional_change = ((current_data['institutional_flows']['total'] - 
                                       last_week['institutional_flows']['total']) / 
                                      last_week['institutional_flows']['total'] * 100)
                
                metrics['retail_flow_change'] = retail_change
                metrics['institutional_flow_change'] = institutional_change
                
                # Determine flow signals
                metrics['retail_signal'] = 'POSITIVE' if retail_change > 5 else 'NEGATIVE' if retail_change < -5 else 'NEUTRAL'
                metrics['institutional_signal'] = 'POSITIVE' if institutional_change > 5 else 'NEGATIVE' if institutional_change < -5 else 'NEUTRAL'
        
        return metrics
    
    def generate_ai_analysis(self, data):
        """Generate AI analysis using Claude"""
        try:
            prompt = f"""
            Weekly Market Flow Analysis - {datetime.now().strftime('%Y-%m-%d')}
            
            Flow Data:
            - Retail Flows: ${data['ici_data']['retail_flows']['total']:,.0f}
            - Institutional Flows: ${data['ici_data']['institutional_flows']['total']:,.0f}
            - Retail Flow Change: {data['metrics'].get('retail_flow_change', 0):.2f}%
            - Institutional Flow Change: {data['metrics'].get('institutional_flow_change', 0):.2f}%
            
            Market Indices:
            - S&P 500: {data['indices'].get('SP500', 'N/A')}
            - ASX 200: {data['indices'].get('ASX200', 'N/A')}
            
            Economic Indicators:
            - Gold: ${data['indicators'].get('gold', 'N/A')}
            - Oil: ${data['indicators'].get('oil', 'N/A')}
            - Yield Curve: {data['indicators'].get('yield_curve', 'N/A')}
            
            Please provide a concise analysis covering:
            1. Key flow trends and market implications
            2. Notable rate of change signals
            3. Economic indicator correlations
            4. Risk factors and opportunities
            5. Outlook for the week ahead
            
            Keep the analysis under 400 words and focus on actionable insights.
            """
            
            response = self.anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"AI analysis unavailable: {str(e)}"
    
    def save_data(self, data):
        """Save data to pickle file"""
        try:
            # Load existing data
            if os.path.exists(self.data_file):
                with open(self.data_file, 'rb') as f:
                    historical = pickle.load(f)
            else:
                historical = []
            
            # Append new data
            historical.append(data)
            
            # Keep only last 52 weeks (1 year)
            if len(historical) > 52:
                historical = historical[-52:]
            
            # Save updated data
            with open(self.data_file, 'wb') as f:
                pickle.dump(historical, f)
                
            return historical
            
        except Exception as e:
            print(f"Error saving data: {e}")
            return []
    
    def send_email_alert(self, analysis, data):
        """Send weekly email with analysis"""
        try:
            sender = self.config['SENDER_EMAIL']
            password = self.config['SENDER_PASSWORD']
            
            for recipient in self.recipients:
                msg = MIMEMultipart('alternative')
                msg['Subject'] = f"Weekly Market Flow Report - {datetime.now().strftime('%Y-%m-%d')}"
                msg['From'] = sender
                msg['To'] = recipient
                
                # Create HTML email
                html = f"""
                <html>
                  <head>
                    <style>
                      body {{ font-family: Arial, sans-serif; }}
                      .header {{ background-color: #2c3e50; color: white; padding: 20px; }}
                      .content {{ padding: 20px; }}
                      .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                      .metric-box {{ 
                        background: #f8f9fa; 
                        padding: 15px; 
                        border-radius: 8px;
                        text-align: center;
                      }}
                      .signal-positive {{ color: green; font-weight: bold; }}
                      .signal-negative {{ color: red; font-weight: bold; }}
                      .signal-neutral {{ color: gray; }}
                      .analysis {{ 
                        background: #e9ecef; 
                        padding: 20px; 
                        border-radius: 8px;
                        margin: 20px 0;
                      }}
                    </style>
                  </head>
                  <body>
                    <div class="header">
                      <h1>📈 Weekly Market Flow Report</h1>
                      <p>{datetime.now().strftime('%B %d, %Y')}</p>
                    </div>
                    
                    <div class="content">
                      <h2>Key Metrics</h2>
                      <div class="metrics">
                        <div class="metric-box">
                          <h3>Retail Flow Change</h3>
                          <p class="signal-{data['metrics'].get('retail_signal', '').lower()}">
                            {data['metrics'].get('retail_flow_change', 0):.2f}%
                          </p>
                        </div>
                        <div class="metric-box">
                          <h3>Institutional Flow Change</h3>
                          <p class="signal-{data['metrics'].get('institutional_signal', '').lower()}">
                            {data['metrics'].get('institutional_flow_change', 0):.2f}%
                          </p>
                        </div>
                        <div class="metric-box">
                          <h3>S&P 500</h3>
                          <p>{data['indices'].get('SP500', 'N/A'):.2f}</p>
                        </div>
                        <div class="metric-box">
                          <h3>ASX 200</h3>
                          <p>{data['indices'].get('ASX200', 'N/A'):.2f}</p>
                        </div>
                      </div>
                      
                      <h2>AI Analysis</h2>
                      <div class="analysis">
                        {analysis.replace(chr(10), '<br>')}
                      </div>
                      
                      <p style="margin-top: 30px; color: #666;">
                        This is an automated weekly report. Visit the dashboard for interactive charts and detailed analysis.
                      </p>
                    </div>
                  </body>
                </html>
                """
                
                part = MIMEText(html, 'html')
                msg.attach(part)
                
                # Send email
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                    server.login(sender, password)
                    server.sendmail(sender, recipient, msg.as_string())
                
                print(f"Email sent to {recipient}")
                
        except Exception as e:
            print(f"Error sending email: {e}")
    
    def weekly_update(self):
        """Main weekly update function"""
        print(f"Starting weekly update - {datetime.now()}")
        
        # Fetch all data
        ici_data = self.fetch_ici_weekly_data()
        if not ici_data:
            print("Failed to fetch ICI data, skipping update")
            return
        
        indices = self.fetch_market_indices()
        indicators = self.fetch_economic_indicators()
        
        # Load historical data
        historical = []
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                historical = pickle.load(f)
        
        # Calculate metrics
        metrics = self.calculate_flow_metrics(ici_data, historical)
        
        # Package data
        weekly_data = {
            'date': datetime.now(),
            'ici_data': ici_data,
            'indices': indices,
            'indicators': indicators,
            'metrics': metrics
        }
        
        # Generate AI analysis
        analysis = self.generate_ai_analysis(weekly_data)
        weekly_data['analysis'] = analysis
        
        # Save data
        self.save_data(weekly_data)
        
        # Send email alerts
        if self.recipients:
            self.send_email_alert(analysis, weekly_data)
        
        print(f"Weekly update completed - {datetime.now()}")

def main():
    """Main scheduler function"""
    print("Starting Market Flow Tracker Scheduler")
    
    # Initialize updater
    updater = WeeklyMarketUpdater()
    
    # Schedule weekly update - ICI data typically releases Wednesday afternoon
    schedule.every().wednesday.at("16:00").do(updater.weekly_update)
    
    # Also run on startup
    updater.weekly_update()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()
