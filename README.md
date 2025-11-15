# Stock Market Flow Tracker - USA & Australia

A comprehensive Streamlit application for tracking retail and institutional investment flows in USA and Australian markets, with economic indicators overlay and AI-powered weekly analysis.

## Features

### Core Functionality
- **Dual Market Tracking**: Separate charts for USA (S&P 500) and Australia (ASX 200)
- **Investment Flow Analysis**: 
  - Retail investor flows
  - Institutional investor flows
  - Rate of change calculations with signal markers
- **Time Range Selection**: 1M, 3M, 6M, 1Y, 3Y, 5Y, 10Y, 15Y, All-time views
- **Weekly Auto-Updates**: Automatic data refresh after ICI weekly releases
- **Email Alerts**: Weekly analysis sent to configured recipients

### Economic Indicators (Toggleable Overlays)
- Gold Prices (FRED)
- Copper Prices (Alpha Vantage)
- Oil Prices - WTI (FRED)
- Yield Curve 10Y-2Y (FRED)
- Credit Spreads (FRED)
- Money Market Funds (FRED)
- Initial Jobless Claims (FRED)
- Mortgage Rates (FRED)
- Consumer Confidence (FRED)
- Options Trading Volume

### AI Analysis
- Weekly market analysis powered by Claude AI
- Historical analysis archive with date selection
- Key trend identification and market outlook
- Rate of change signal interpretation

## Installation

### Prerequisites
- Python 3.8 or higher
- Streamlit account (for deployment)
- API keys (see Configuration section)

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/market-flow-tracker.git
cd market-flow-tracker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create Streamlit secrets directory:
```bash
mkdir .streamlit
```

4. Configure API keys (see Configuration section)

5. Run the application:
```bash
streamlit run stock_market_app.py
```

### Deployment to Streamlit Cloud

1. Push code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Configure secrets in Streamlit Cloud dashboard
5. Deploy

## Configuration

### Required API Keys

Create `.streamlit/secrets.toml` file with the following:

```toml
# FRED API Key (Free)
# Get from: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = "your_fred_api_key"

# Alpha Vantage API Key (Free)
# Get from: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_KEY = "your_alpha_vantage_key"

# Anthropic Claude API Key
# Get from: https://console.anthropic.com/
ANTHROPIC_API_KEY = "your_anthropic_api_key"

# ICI API Key (Subscription required)
# Get from: https://www.ici.org/
ICI_API_KEY = "your_ici_api_key"

# Email Configuration (for alerts)
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_specific_password"
```

### Email Setup (Gmail)

1. Enable 2-factor authentication in Gmail
2. Generate app-specific password:
   - Go to Google Account settings
   - Security → 2-Step Verification → App passwords
   - Generate password for "Mail"
3. Use this password in `SENDER_PASSWORD`

## Usage

### Main Dashboard

1. **Select Market View**: Choose USA, Australia, or Both
2. **Time Range**: Select desired time period
3. **Toggle Indicators**: Check/uncheck economic indicators to overlay
4. **Update Data**: Click to fetch latest data
5. **View Analysis**: Read current and historical AI analyses

### Rate of Change Signals

- **Green Triangle Up (▲)**: Significant positive rate change (>90th percentile)
- **Red Triangle Down (▼)**: Significant negative rate change (<10th percentile)
- Signals indicate potential market turning points

### Weekly Scheduler

To run automatic weekly updates:

```bash
python weekly_scheduler.py
```

This will:
- Fetch ICI data every Wednesday at 4 PM
- Generate AI analysis
- Send email alerts to configured recipients
- Update cached data

## Data Sources

- **ICI (Investment Company Institute)**: Weekly mutual fund flow data
- **FRED (Federal Reserve Economic Data)**: Economic indicators
- **Alpha Vantage**: Commodity prices (copper)
- **Yahoo Finance**: Stock market indices
- **Anthropic Claude**: AI analysis

## File Structure

```
market-flow-tracker/
├── stock_market_app.py       # Main Streamlit application
├── weekly_scheduler.py        # Automated update scheduler
├── requirements.txt           # Python dependencies
├── config_template.toml       # Configuration template
├── README.md                  # Documentation
├── .streamlit/
│   └── secrets.toml          # API keys (local only)
└── data/
    ├── market_data.pkl       # Cached market data
    └── analysis_history.pkl  # Historical AI analyses
```

## Customization

### Adding New Indicators

1. Add data fetching logic in `fetch_economic_indicators()`
2. Add checkbox in sidebar configuration
3. Include in `selected_indicators` dict
4. Add to chart creation logic

### Modifying Email Templates

Edit the HTML template in `send_email_alert()` function in `weekly_scheduler.py`

### Changing Update Schedule

Modify the schedule in `weekly_scheduler.py`:
```python
schedule.every().wednesday.at("16:00").do(updater.weekly_update)
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Verify all keys are correctly set in secrets.toml
2. **Data Not Loading**: Check internet connection and API rate limits
3. **Email Not Sending**: Verify Gmail app-specific password is correct
4. **Charts Not Displaying**: Ensure Plotly is installed correctly

### Rate Limits

- FRED API: 120 requests per minute
- Alpha Vantage: 5 requests per minute (free tier)
- Yahoo Finance: No official limit but use responsibly

## Cost Estimates

- **Anthropic Claude API**: ~$1.56/year (52 weekly analyses)
- **Other APIs**: Free tiers sufficient for normal usage
- **Streamlit Cloud**: Free for public apps

## Support

For issues or questions:
1. Check the documentation
2. Review troubleshooting section
3. Open an issue on GitHub

## License

MIT License - See LICENSE file for details

## Disclaimer

This tool is for informational purposes only. Not financial advice. Always conduct your own research before making investment decisions.

## Future Enhancements

- [ ] Add real-time ASX flow data integration
- [ ] Include sentiment analysis from financial forums
- [ ] Add portfolio tracking features
- [ ] Implement backtesting capabilities
- [ ] Create mobile app version
- [ ] Add more international markets
- [ ] Include cryptocurrency flow tracking
