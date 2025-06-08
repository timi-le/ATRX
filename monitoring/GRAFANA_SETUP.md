# FX Quant Trading System - Grafana Monitoring Setup

## 🎯 Overview

This is the complete Grafana monitoring stack for the FX Quant Trading System (Task 20). It provides real-time visualization of trading performance, system health, execution quality, and risk metrics through professional dashboards.

## 📦 Components

- **Grafana**: Visualization and dashboarding (Port 3000)
- **Prometheus**: Metrics collection and storage (Port 9090)
- **Alertmanager**: Alert routing and notification (Port 9093)
- **FX Metrics Server**: Custom metrics from trading system (Port 9000)

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Run the automated setup script
python setup_grafana_monitoring.py

# Validate the installation
python test_grafana_setup.py
```

### Option 2: Manual Setup

```bash
# 1. Start the monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# 2. Start the metrics server (from Task 19)
python demo_metrics_monitoring.py

# 3. Access Grafana at http://localhost:3000 (admin/admin)
# 4. Import dashboards manually from monitoring/grafana/dashboards/
```

## 🌐 Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | admin/admin |
| Prometheus | http://localhost:9090 | None |
| Alertmanager | http://localhost:9093 | None |
| Metrics Endpoint | http://localhost:9000/metrics | None |

## 📊 Available Dashboards

### 1. FX AI-Quant Trading Performance
- **Purpose**: Main trading dashboard with comprehensive performance metrics
- **Key Panels**:
  - Real-time equity curve (Total Equity, Daily PnL)
  - Execution latency gauge (< 100ms threshold)
  - Market regime distribution (pie chart)
  - Trade statistics (hourly count, win rate)
  - System error monitoring
  - Current drawdown gauge
  - Sharpe ratio indicator
  - CPU usage monitoring
  - Position exposure by symbol
  - Trade execution rate

### 2. FX AI-Quant System Health
- **Purpose**: Technical system monitoring and diagnostics
- **Key Panels**:
  - System resource usage (CPU, Memory)
  - Latency metrics (Execution, Fill, Market Data)
  - Error and exception rates
  - Market data update rates
  - Current resource gauges
  - Fill rate monitoring
  - System information table

## 🔔 Alerting Rules

### Critical Alerts (Immediate Response Required)
- **High Drawdown**: > 5% drawdown for 2 minutes
- **System Down**: Trading system offline for 30 seconds
- **Excessive Errors**: > 0.1 errors/second for 1 minute
- **Excessive Drawdown**: > 10% maximum drawdown

### Warning Alerts (Review Required)
- **High Execution Latency**: > 100ms for 2 minutes
- **Low Fill Rate**: < 80% for 3 minutes
- **High Slippage**: > 3 pips average for 2 minutes
- **Market Data Lag**: > 0.5 seconds for 1 minute
- **Low Win Rate**: < 40% for 10 minutes
- **High CPU/Memory**: > 85% CPU or > 2GB memory for 5 minutes
- **Strategy Underperformance**: < -$1000 PnL for 5 minutes
- **Low Sharpe Ratio**: < 0.5 for 15 minutes

### Info Alerts (Awareness)
- **High Position Exposure**: > $500K total exposure
- **No Recent Trades**: No trades for 10 minutes
- **Regime Change**: > 30% regime distribution change
- **Significant Daily PnL**: > $5000 profit or < -$2000 loss

## ⚙️ Configuration

### Docker Environment Variables

```yaml
# Grafana Configuration
GF_SECURITY_ADMIN_PASSWORD: admin
GF_INSTALL_PLUGINS: grafana-piechart-panel
GF_USERS_ALLOW_SIGN_UP: false

# Prometheus Configuration
- Port: 9090
- Scrape Interval: 5 seconds
- Data Retention: 200 hours

# Alertmanager Configuration
- Port: 9093
- Route Grouping: By alertname and component
- Repeat Interval: 1 hour (critical: 5 minutes)
```

### Prometheus Targets

```yaml
job_name: 'fx-ai-trading-system'
static_configs:
  - targets: ['host.docker.internal:9000']
scrape_interval: 5s
metrics_path: /metrics
```

### Email/Slack Notifications

Edit `monitoring/alertmanager.yml`:

```yaml
global:
  smtp_smarthost: 'your-smtp-server:587'
  smtp_from: 'fx-trading-alerts@yourcompany.com'

receivers:
  - name: 'critical-alerts'
    email_configs:
      - to: 'trading-team@yourcompany.com'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#fx-trading-alerts'
```

## 🧪 Testing and Validation

### Automated Testing

```bash
# Run comprehensive test suite
python test_grafana_setup.py

# Expected output:
# ✅ Docker Containers: PASSED
# ✅ Metrics Server: PASSED
# ✅ Prometheus: PASSED
# ✅ Grafana: PASSED
# ✅ Grafana Datasource: PASSED
# ✅ Grafana Dashboards: PASSED
# ✅ Alertmanager: PASSED
# ✅ Sample Data Generation: PASSED
```

### Manual Validation

1. **Check Docker Containers**:
   ```bash
   docker ps | grep -E "(grafana|prometheus|alertmanager)"
   ```

2. **Verify Metrics Endpoint**:
   ```bash
   curl http://localhost:9000/metrics | grep fxai_
   ```

3. **Test Prometheus Query**:
   ```bash
   curl "http://localhost:9090/api/v1/query?query=fxai_pnl_equity"
   ```

4. **Access Grafana Dashboard**:
   - Visit http://localhost:3000
   - Login with admin/admin
   - Navigate to dashboards

## 🔧 Troubleshooting

### Common Issues

#### 1. Docker Containers Not Starting

```bash
# Check logs
docker-compose -f docker-compose.monitoring.yml logs

# Common solutions:
- Ensure ports 3000, 9090, 9093 are available
- Check Docker daemon is running
- Restart Docker service
```

#### 2. Prometheus Can't Scrape Metrics

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Solutions:
- Ensure metrics server is running on port 9000
- Check firewall settings
- Verify host.docker.internal resolution
```

#### 3. Grafana Dashboards Not Loading

```bash
# Check Grafana logs
docker logs fx-grafana

# Solutions:
- Verify Prometheus datasource is configured
- Re-import dashboards manually
- Check dashboard JSON syntax
```

#### 4. No Metrics Data Visible

```bash
# Start metrics server with sample data
python demo_metrics_monitoring.py

# Or generate sample data programmatically
python -c "
from monitoring.metrics_server import get_metrics
m = get_metrics()
m.update_pnl(equity=100000, daily_pnl=1000)
m.record_trade('EURUSD', 'BUY', 'filled', pnl=250)
"
```

### Advanced Debugging

#### Enable Debug Logging

```bash
# Grafana debug mode
docker-compose -f docker-compose.monitoring.yml up -d --build
docker exec -it fx-grafana grafana-cli admin reset-admin-password admin
```

#### Prometheus Query Testing

```bash
# Test specific metrics
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=fxai_pnl_equity' \
  --data-urlencode 'time=2023-01-01T00:00:00Z'
```

#### Alertmanager Testing

```bash
# Send test alert
curl -XPOST http://localhost:9093/api/v1/alerts \
  -H "Content-Type: application/json" \
  -d '[{"labels":{"alertname":"TestAlert","severity":"warning"}}]'
```

## 🔌 Integration with Trading System

### Step 1: Import Metrics Server

```python
from monitoring.metrics_server import get_metrics

# Get global metrics instance
metrics = get_metrics()
```

### Step 2: Record Trading Events

```python
# Record trade execution
metrics.record_trade(
    symbol="EURUSD",
    side="BUY",
    status="filled",
    pnl=150.25
)

# Update position exposure
metrics.update_position_exposure(
    symbol="EURUSD",
    side="long",
    exposure_usd=50000.0
)

# Record execution metrics
metrics.record_execution_latency(0.025)  # 25ms
metrics.record_slippage(1.2)  # 1.2 pips
```

### Step 3: Update Performance Metrics

```python
# Update PnL tracking
metrics.update_pnl(
    equity=105000.0,
    daily_pnl=5000.0,
    unrealized_pnl=500.0
)

# Update risk metrics
metrics.update_risk_metrics(
    current_drawdown=0.015,  # 1.5%
    max_drawdown=0.025       # 2.5%
)
```

### Step 4: Monitor System Health

```python
# Update system resources
metrics.update_system_resources(
    memory_mb=756.2,
    cpu_percent=35.5
)

# Record errors
metrics.record_error(
    error_type="connection_timeout",
    component="broker_api"
)
```

## 📈 Performance Optimization

### Prometheus Optimization

```yaml
# prometheus.yml optimizations
global:
  scrape_interval: 5s      # Reduce for real-time trading
  evaluation_interval: 5s  # Fast alert evaluation

storage:
  tsdb:
    retention.time: 7d     # Adjust based on disk space
    retention.size: 10GB   # Prevent disk overflow
```

### Grafana Optimization

```yaml
# Enable panel query caching
GF_PANELS_DISABLE_SANITIZE_HTML: true
GF_SECURITY_ALLOW_EMBEDDING: true

# Increase query timeout for complex queries
GF_DATABASE_QUERY_TIMEOUT: 60s
```

### Metrics Server Optimization

```python
# Use efficient metric updates
metrics.batch_update([
    ("pnl_equity", 105000.0),
    ("daily_pnl", 5000.0),
    ("cpu_percent", 35.5)
])
```

## 🛡️ Security Considerations

### 1. Change Default Passwords

```bash
# Change Grafana admin password
docker exec -it fx-grafana grafana-cli admin reset-admin-password YOUR_SECURE_PASSWORD
```

### 2. Enable HTTPS

```yaml
# docker-compose.monitoring.yml
grafana:
  environment:
    - GF_SERVER_PROTOCOL=https
    - GF_SERVER_CERT_FILE=/etc/ssl/certs/grafana.crt
    - GF_SERVER_CERT_KEY=/etc/ssl/private/grafana.key
```

### 3. Restrict Network Access

```yaml
# Bind to localhost only
ports:
  - "127.0.0.1:3000:3000"  # Grafana
  - "127.0.0.1:9090:9090"  # Prometheus
  - "127.0.0.1:9093:9093"  # Alertmanager
```

## 📝 Maintenance

### Regular Tasks

1. **Weekly**: Review alert thresholds and update based on trading patterns
2. **Monthly**: Clean up old Prometheus data and optimize queries
3. **Quarterly**: Update dashboard layouts and add new metrics

### Backup Strategy

```bash
# Backup Grafana dashboards
curl -u admin:admin http://localhost:3000/api/search > dashboards_backup.json

# Backup Prometheus data
docker exec fx-prometheus tar -czf /prometheus_backup.tar.gz /prometheus/data

# Backup alerting configuration
cp monitoring/alert_rules.yml monitoring/backup/alert_rules_$(date +%Y%m%d).yml
```

## 🎓 Best Practices

1. **Dashboard Design**:
   - Use consistent color schemes
   - Include contextual legends
   - Set appropriate time ranges
   - Add helpful annotations

2. **Alerting Strategy**:
   - Start with high thresholds and adjust down
   - Use different severities appropriately
   - Include runbook links in alerts
   - Test alert routing regularly

3. **Metrics Naming**:
   - Use consistent prefixes (fxai_)
   - Include units in descriptions
   - Add meaningful labels
   - Document custom metrics

4. **Performance Monitoring**:
   - Monitor dashboard load times
   - Optimize slow queries
   - Use downsampling for historical data
   - Cache frequently accessed metrics

## 🤝 Support

For issues related to this monitoring setup:

1. Check the troubleshooting section above
2. Run the diagnostic script: `python test_grafana_setup.py`
3. Review Docker container logs
4. Verify metrics server is producing data

## 📚 Additional Resources

- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [PromQL Query Language](https://prometheus.io/docs/prometheus/latest/querying/)
- [Alertmanager Configuration](https://prometheus.io/docs/alerting/latest/alertmanager/)

---

**Task 20: Dashboard and Visualization - Complete** ✅

This monitoring stack provides production-ready visibility into your FX trading system performance and health metrics.
