# Mystic Trading Platform - Streamlit Dashboard

A real-time, interactive dashboard for the Mystic Trading Platform built with Streamlit. This dashboard provides live market data, portfolio tracking, performance analytics, and AI strategy monitoring.

## 🚀 Features

### 📈 Overview Page
- **Live Portfolio Performance**: Real-time portfolio value tracking with historical charts
- **Market Metrics**: Live market data including tracked symbols and market trends
- **Connection Status**: Real-time backend connectivity monitoring
- **Auto-refresh**: Automatic data updates every 30 seconds

### 📊 Trading Performance Page
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate, profit factor
- **Trade Distribution**: Visual breakdown of trade types and strategies
- **Live Data**: All metrics pulled from actual trading data

### 🤖 AI Strategies Page
- **Strategy Performance**: Real-time performance of active AI strategies
- **AI Insights**: Live AI-generated trading insights and recommendations
- **Strategy Evolution**: Monitoring of AI learning and adaptation

### 💼 Portfolio Page
- **Asset Allocation**: Live portfolio allocation charts
- **Current Holdings**: Real-time position tracking with prices and 24h changes
- **Portfolio Analytics**: Comprehensive portfolio performance metrics

### ⚙️ Settings Page
- **Display Settings**: Customize dashboard appearance and behavior
- **Trading Settings**: Configure position sizing, stop losses, and take profits
- **AI Settings**: Enable/disable AI trading features

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Backend server running on port 8000
- Required Python packages (see requirements_streamlit.txt)

### Quick Start

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements_streamlit.txt
   ```

3. **Start the backend server** (if not already running):
   ```bash
   cd ../backend
   python main.py
   ```

4. **Launch the dashboard**:

   **Option A: Using PowerShell (Recommended)**
   ```powershell
   .\run_streamlit_dashboard.ps1
   ```

   **Option B: Using Batch file**
   ```cmd
   run_streamlit_dashboard.bat
   ```

   **Option C: Direct command**
   ```bash
   streamlit run streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0
   ```

5. **Access the dashboard**:
   Open your browser and go to: `http://localhost:8501`

## 🔧 Configuration

### Backend Connection
The dashboard automatically connects to the backend at `http://localhost:8000`. If your backend is running on a different port or host, modify the `BACKEND_URL` variable in `streamlit_dashboard.py`.

### Auto-refresh Settings
- **Default**: 30 seconds
- **Cache Duration**: 30-60 seconds (varies by data type)
- **Manual Refresh**: Click the "🔄 Refresh Data" button

### Data Sources
- **Market Data**: CoinGecko API (live cryptocurrency prices)
- **Portfolio Data**: Backend portfolio service
- **Performance Data**: Backend analytics service
- **AI Insights**: Backend AI service

## 📊 Data Flow

```
Streamlit Dashboard
        ↓
   HTTP Requests
        ↓
   Backend API (Port 8000)
        ↓
   Live Market Data Service
        ↓
   CoinGecko API / Exchange APIs
```

## 🚨 Troubleshooting

### Common Issues

1. **Backend Connection Failed**
   - Ensure the backend server is running on port 8000
   - Check firewall settings
   - Verify the backend health endpoint: `http://localhost:8000/health`

2. **No Data Displayed**
   - Check the connection status in the sidebar
   - Verify API endpoints are responding
   - Run the test script: `python test_backend_connection.py`

3. **Slow Performance**
   - Reduce auto-refresh frequency
   - Check network connectivity
   - Monitor backend server performance

### Testing Backend Connection

Run the included test script to verify all endpoints:
```bash
python test_backend_connection.py
```

This will test all endpoints used by the dashboard and provide a detailed report.

## 📈 API Endpoints Used

The dashboard connects to these backend endpoints:

- `/health` - Backend health check
- `/api/live/market-data` - Live market data
- `/api/portfolio/overview` - Portfolio summary
- `/api/portfolio/positions` - Portfolio positions
- `/api/analytics/performance` - Performance metrics
- `/api/analytics/trade-history` - Trade history
- `/api/analytics/strategies` - Strategy performance
- `/api/analytics/ai-insights` - AI insights
- `/live/global` - Global market data

## 🔒 Security Notes

- The dashboard runs on `0.0.0.0:8501` by default (accessible from any IP)
- For production use, consider restricting access with firewall rules
- API keys and sensitive data are handled by the backend, not the dashboard
- All data is cached locally for 30-60 seconds to reduce API calls

## 📝 Development

### Adding New Pages
1. Add the page name to the sidebar selectbox
2. Create a new `elif` block for the page
3. Add appropriate data fetching functions
4. Implement the UI components

### Adding New Data Sources
1. Create a new cached function in the dashboard
2. Add the corresponding backend endpoint
3. Update the data processing logic
4. Test with the connection test script

### Customization
- Modify the `BACKEND_URL` for different environments
- Adjust cache durations for different data types
- Customize the UI theme and layout
- Add new metrics and visualizations

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Run the backend connection test
3. Check backend logs for errors
4. Verify all dependencies are installed

## 🎯 Performance Tips

- Use the cache decorators for expensive API calls
- Implement error handling for all data fetching
- Monitor backend response times
- Consider implementing websockets for real-time updates
- Use appropriate chart types for different data sets

---

**Note**: This dashboard requires a running backend server with live market data capabilities. Ensure all backend services are properly configured and running before launching the dashboard. 