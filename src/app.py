from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import io
import base64
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='.')
CORS(app)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Google Sheets Configuration
SPREADSHEET_ID = '1swNSIGLsCL-O_31tXhPXeyDoMntUin3hMyxTLVLvnr0'
SHEET_NAME = 'Sheet1'

# Global variables
cached_data = None
last_fetch_time = None
CACHE_DURATION = 60

class EnergyAnalysisSystem:
    """Complete Energy Analysis System - ALL FEATURES FROM ORIGINAL"""
    
    def __init__(self):
        self.df = None
        self.models = {}
        
    def fetch_data_from_sheets(self):
        """Fetch data from Google Sheets"""
        try:
            sheet_url = f'https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv&gid=0'
            df = pd.read_csv(sheet_url)
            df.columns = df.columns.str.strip()
            
            # Ensure required columns
            required_cols = ['Voltage', 'Current', 'Power']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Convert to numeric
            df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
            df['Current'] = pd.to_numeric(df['Current'], errors='coerce')
            df['Power'] = pd.to_numeric(df['Power'], errors='coerce')
            df = df.dropna(subset=['Voltage', 'Current', 'Power'])
            
            # Add timestamp
            if 'Timestamp' not in df.columns:
                start_time = datetime.now() - timedelta(seconds=len(df) * 6)
                df['Timestamp'] = [start_time + timedelta(seconds=i*6) for i in range(len(df))]
            else:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            
            self.df = df.sort_values('Timestamp')
            self.prepare_features()
            
            print(f"✓ Loaded {len(df)} data points from Google Sheets")
            return df
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def prepare_features(self):
        """Extract time-based features for ML models"""
        self.df['Hour'] = self.df['Timestamp'].dt.hour
        self.df['DayOfWeek'] = self.df['Timestamp'].dt.dayofweek
        self.df['Month'] = self.df['Timestamp'].dt.month
        self.df['DayOfMonth'] = self.df['Timestamp'].dt.day
        self.df['IsWeekend'] = (self.df['DayOfWeek'] >= 5).astype(int)
        
        # Rolling averages
        self.df['Power_MA_5'] = self.df['Power'].rolling(window=5, min_periods=1).mean()
        self.df['Power_MA_10'] = self.df['Power'].rolling(window=10, min_periods=1).mean()
        self.df['Voltage_Stability'] = self.df['Voltage'].rolling(window=5, min_periods=1).std()
        
        # Power factor and efficiency
        self.df['PowerFactor'] = self.df['Power'] / (self.df['Voltage'] * self.df['Current'])
        self.df['PowerFactor'] = self.df['PowerFactor'].clip(0, 1)
        self.df['Efficiency'] = self.df['PowerFactor'] * 100
    
    def ai_energy_prediction(self, hours_ahead=24):
        """AI-powered energy prediction using ensemble methods"""
        if self.df is None or len(self.df) < 50:
            return None
        
        try:
            features = ['Hour', 'DayOfWeek', 'Month', 'Voltage', 'Current', 
                       'Power_MA_5', 'Power_MA_10', 'Voltage_Stability']
            X = self.df[features].fillna(method='bfill')
            y = self.df['Power']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            
            rf_model.fit(X_train_scaled, y_train)
            gb_model.fit(X_train_scaled, y_train)
            
            rf_score = rf_model.score(X_test_scaled, y_test)
            gb_score = gb_model.score(X_test_scaled, y_test)
            
            # Future predictions
            last_timestamp = self.df['Timestamp'].max()
            predictions = []
            
            for i in range(1, hours_ahead + 1):
                ts = last_timestamp + timedelta(hours=i)
                hour_features = np.array([[
                    ts.hour, ts.weekday(), ts.month,
                    self.df['Voltage'].mean(),
                    self.df['Current'].mean(),
                    self.df['Power'].tail(5).mean(),
                    self.df['Power'].tail(10).mean(),
                    self.df['Voltage'].tail(5).std()
                ]])
                hour_features_scaled = scaler.transform(hour_features)
                
                rf_p = rf_model.predict(hour_features_scaled)[0]
                gb_p = gb_model.predict(hour_features_scaled)[0]
                pred_power = (rf_p + gb_p) / 2
                
                predictions.append({
                    'timestamp': ts.strftime('%Y-%m-%d %H:%M'),
                    'hour': ts.strftime('%H:%M'),
                    'predicted_power': float(pred_power),
                    'confidence': float(min(rf_score, gb_score) * 100)
                })
            
            return {
                'predictions': predictions,
                'model_performance': {
                    'rf_score': float(rf_score),
                    'gb_score': float(gb_score),
                    'ensemble_score': float((rf_score + gb_score) / 2)
                }
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def batch_efficiency_analysis(self):
        """Batch processing for efficiency patterns"""
        if self.df is None:
            return None
        
        voltage_std = self.df['Voltage'].std()
        low_efficiency = self.df[self.df['Efficiency'] < 95]
        energy_waste = low_efficiency['Power'].sum() * 0.05 / 60
        
        return {
            'avgPowerFactor': float(self.df['PowerFactor'].mean()),
            'avgEfficiency': float(self.df['Efficiency'].mean()),
            'peakEfficiency': float(self.df['Efficiency'].max()),
            'minEfficiency': float(self.df['Efficiency'].min()),
            'voltageRange': {
                'min': float(self.df['Voltage'].min()),
                'max': float(self.df['Voltage'].max()),
                'std': float(voltage_std)
            },
            'stabilityRating': 'Excellent' if voltage_std < 5 else 'Good' if voltage_std < 10 else 'Fair',
            'lowEfficiencyPeriods': len(low_efficiency),
            'energyWaste': float(energy_waste),
            'potentialSavings': float(energy_waste * 0.008)
        }
    
    def monthly_energy_report(self):
        """Generate comprehensive monthly energy report"""
        if self.df is None:
            return None
        
        total_energy = self.df['Power'].sum() / 60 / 1000
        cost_per_kwh = 8
        monthly_cost = total_energy * cost_per_kwh
        
        hourly_stats = self.df.groupby('Hour')['Power'].agg(['mean', 'max', 'min'])
        
        return {
            'totalEnergy': float(total_energy),
            'avgPower': float(self.df['Power'].mean()),
            'peakPower': float(self.df['Power'].max()),
            'minPower': float(self.df['Power'].min()),
            'monthlyCost': float(monthly_cost),
            'dailyAvgCost': float(monthly_cost / 30),
            'hourlyPattern': {
                'hours': hourly_stats.index.tolist(),
                'mean': hourly_stats['mean'].tolist(),
                'max': hourly_stats['max'].tolist(),
                'min': hourly_stats['min'].tolist()
            }
        }
    
    def detect_peak_hours(self):
        """Advanced peak hour detection"""
        if self.df is None:
            return None
        
        hourly_avg = self.df.groupby('Hour')['Power'].mean()
        
        if len(hourly_avg) < 5:
            threshold = hourly_avg.mean() + hourly_avg.std() * 0.5
        else:
            threshold = hourly_avg.mean() + hourly_avg.std()
        
        peak_hours = hourly_avg[hourly_avg > threshold]
        
        if len(peak_hours) == 0:
            peak_hours = hourly_avg.nlargest(min(3, len(hourly_avg)))
        
        peak_hour_data = self.df[self.df['Hour'].isin(peak_hours.index)]['Power']
        offpeak_hour_data = self.df[~self.df['Hour'].isin(peak_hours.index)]['Power']
        
        peak_cost = 0
        offpeak_cost = 0
        
        if len(peak_hour_data) > 0 and len(offpeak_hour_data) > 0:
            peak_cost = sum(peak_hour_data) / 60000 * 12
            offpeak_cost = sum(offpeak_hour_data) / 60000 * 6
        else:
            total_consumption = sum(self.df['Power']) / 60000
            peak_cost = total_consumption * 0.4 * 12
            offpeak_cost = total_consumption * 0.6 * 6
        
        return {
            'peakHours': peak_hours.index.tolist(),
            'peakPowers': peak_hours.values.tolist(),
            'threshold': float(threshold),
            'average': float(hourly_avg.mean()),
            'baseLoad': float(hourly_avg.min()),
            'peakLoad': float(hourly_avg.max()),
            'loadFactor': float(hourly_avg.mean() / hourly_avg.max()),
            'peakCost': float(peak_cost),
            'offPeakCost': float(offpeak_cost),
            'potentialSavings': float(peak_cost * 0.3),
            'hourlyData': {
                'hours': hourly_avg.index.tolist(),
                'powers': hourly_avg.values.tolist()
            }
        }
    
    def appliance_breakdown(self):
        """Appliance-level consumption estimation"""
        if self.df is None:
            return None
        
        # K-means clustering
        X_cluster = self.df[['Power', 'Current', 'Voltage']].values
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        self.df['Cluster'] = kmeans.fit_predict(X_cluster)
        
        power_ranges = {
            'Base Load (Standby)': (0, 100),
            'Lighting': (100, 300),
            'Fans/Small Appliances': (300, 600),
            'Computer/TV': (600, 800),
            'Refrigerator/AC': (800, 1000),
            'High Power Equipment': (1000, 1500)
        }
        
        total_power = self.df['Power'].sum()
        appliance_data = []
        
        for category, (min_pow, max_pow) in power_ranges.items():
            subset = self.df[(self.df['Power'] >= min_pow) & (self.df['Power'] < max_pow)]
            if len(subset) > 0:
                appliance_data.append({
                    'category': category,
                    'avgPower': float(subset['Power'].mean()),
                    'energy': float(subset['Power'].sum() / 60),
                    'percentage': float((subset['Power'].sum() / total_power) * 100)
                })
        
        return appliance_data
    
    def carbon_credit_tracking(self):
        """Carbon emissions and credit generation tracking"""
        if self.df is None:
            return None
        
        total_kwh = self.df['Power'].sum() / 60000
        carbon_intensity = 0.82
        total_emissions = total_kwh * carbon_intensity
        
        efficiency_improvement = 0.15
        carbon_saved = total_emissions * efficiency_improvement
        carbon_credits = carbon_saved / 1000
        credit_value = carbon_credits * 1200
        
        solar_offset = total_kwh * 0.95
        trees_equivalent = total_emissions / 21
        
        return {
            'totalEnergy': float(total_kwh),
            'totalEmissions': float(total_emissions),
            'monthlyEmissions': float(total_emissions * 30),
            'annualEmissions': float(total_emissions * 365),
            'annualTonnes': float(total_emissions * 365 / 1000),
            'carbonSaved': float(carbon_saved),
            'carbonCredits': float(carbon_credits),
            'creditValue': float(credit_value),
            'annualCreditValue': float(credit_value * 12),
            'solarOffset': float(solar_offset),
            'treesEquivalent': float(trees_equivalent),
            'solarCapacity': float(solar_offset / 4)
        }
    
    def generate_personalized_recommendations(self, profile, avg_power, peak_power, current_pf):
        """Generate personalized recommendations based on user profile"""
        recommendations = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }
        
        # Building age and insulation
        if profile['building_age'] in ['c', 'd'] and profile['insulation'] in ['b', 'c']:
            recommendations['high_priority'].append({
                'title': 'Thermal Insulation Upgrade',
                'reason': f"Your building is {'>15 years old' if profile['building_age'] == 'c' else '>30 years old'} with inadequate insulation",
                'action': 'Install roof insulation (R-30 rating) and wall insulation. Seal air leaks around doors and windows.',
                'cost': '₹80,000 - ₹1,50,000',
                'savings': '25-30% reduction in cooling costs',
                'payback': '2-3 years'
            })
        
        # Lighting analysis
        if profile['lighting'] == 'c':
            recommendations['high_priority'].append({
                'title': 'Complete LED Conversion',
                'reason': 'Less than 40% of your lighting is LED - massive energy waste',
                'action': 'Replace all incandescent and CFL bulbs with LED. Focus on high-usage areas first.',
                'cost': '₹15,000 - ₹40,000',
                'savings': '75-80% lighting energy savings',
                'payback': '8-12 months'
            })
        elif profile['lighting'] == 'b':
            recommendations['medium_priority'].append({
                'title': 'LED Conversion Completion',
                'reason': '40-80% LED coverage - complete the transition',
                'action': 'Replace remaining non-LED bulbs, especially in frequently used areas.',
                'cost': '₹8,000 - ₹20,000',
                'savings': '40-50% on remaining lighting load',
                'payback': '10-14 months'
            })
        
        # Cooling system
        if profile['cooling_system'] == 'c':
            recommendations['high_priority'].append({
                'title': 'Replace Window AC Units',
                'reason': 'Window AC units are 40% less efficient than modern split ACs',
                'action': 'Replace with 5-star inverter split AC units. Consider one unit per room for better efficiency.',
                'cost': '₹35,000 - ₹50,000 per unit',
                'savings': '40-45% reduction in cooling costs',
                'payback': '3-4 years'
            })
        elif profile['cooling_system'] in ['a', 'b']:
            recommendations['medium_priority'].append({
                'title': 'Smart Thermostat Installation',
                'reason': 'Optimize your existing AC system with intelligent control',
                'action': 'Install Wi-Fi enabled smart thermostats with scheduling and occupancy detection.',
                'cost': '₹5,000 - ₹15,000 per unit',
                'savings': '15-20% on cooling costs',
                'payback': '12-18 months'
            })
        
        # Windows
        if profile['windows'] in ['b', 'c']:
            recommendations['medium_priority'].append({
                'title': 'Window Upgrade or Treatment',
                'reason': 'Single-pane or old windows cause significant heat gain',
                'action': 'Option 1: Install reflective window films (cheaper). Option 2: Upgrade to double-glazed windows (better long-term).',
                'cost': 'Films: ₹100-200/sq.ft, New windows: ₹600-1000/sq.ft',
                'savings': 'Films: 10-15%, New windows: 20-25% cooling savings',
                'payback': 'Films: 2-3 years, Windows: 5-7 years'
            })
        
        # Power factor
        if current_pf < 0.85:
            recommendations['high_priority'].append({
                'title': 'Power Factor Correction - URGENT',
                'reason': f'Your power factor is {current_pf:.3f} - causing excess current draw and penalties',
                'action': 'Install automatic capacitor bank (APFC panel) matched to your load.',
                'cost': '₹25,000 - ₹60,000',
                'savings': f'{((1 - current_pf/0.95) * 100):.1f}% current reduction, avoid PF penalties',
                'payback': '12-18 months'
            })
        elif current_pf < 0.90:
            recommendations['medium_priority'].append({
                'title': 'Power Factor Improvement',
                'reason': f'Power factor of {current_pf:.3f} can be optimized',
                'action': 'Install capacitor bank to bring PF above 0.95',
                'cost': '₹20,000 - ₹40,000',
                'savings': f'{((1 - current_pf/0.95) * 100):.1f}% current reduction',
                'payback': '18-24 months'
            })
        
        # Solar recommendations
        if profile['solar_interest'] == 'a' and profile['budget'] in ['c', 'd']:
            recommended_kw = (avg_power * profile['occupancy_hours']) / 4000
            recommendations['high_priority'].append({
                'title': 'Rooftop Solar Installation',
                'reason': 'High interest + adequate budget + good payback for your consumption pattern',
                'action': f"Install {recommended_kw:.1f} kW rooftop solar system with net metering. Include battery backup for critical loads.",
                'cost': f"₹{recommended_kw * 50000:.0f} - ₹{recommended_kw * 60000:.0f}",
                'savings': f'70-80% reduction in grid dependency, ₹{(recommended_kw * 1500 * 8):.0f}/year',
                'payback': '4-5 years'
            })
        elif profile['solar_interest'] in ['a', 'b']:
            recommended_kw = (avg_power * profile['occupancy_hours']) / 4000
            recommendations['medium_priority'].append({
                'title': 'Solar Energy Feasibility',
                'reason': 'Solar can significantly reduce your electricity costs',
                'action': f"Get quotes for {recommended_kw:.1f} kW solar system. Check for government subsidies (up to 40% for residential).",
                'cost': f"₹{recommended_kw * 50000 * 0.6:.0f} after subsidy - ₹{recommended_kw * 60000:.0f} full price",
                'savings': f'65-75% grid independence, ₹{(recommended_kw * 1500 * 8):.0f}/year',
                'payback': '4-6 years (3-4 years with subsidy)'
            })
        
        # Building type specific
        if profile['building_type'] == 'b':
            recommendations['medium_priority'].append({
                'title': 'Occupancy-Based Automation',
                'reason': 'Commercial buildings benefit greatly from occupancy sensors',
                'action': 'Install PIR sensors for lighting and HVAC control in conference rooms, restrooms, and corridors.',
                'cost': '₹50,000 - ₹1,20,000',
                'savings': '25-35% on lighting and HVAC in low-occupancy areas',
                'payback': '2-3 years'
            })
        elif profile['building_type'] == 'a':
            recommendations['low_priority'].append({
                'title': 'Smart Home Energy Management',
                'reason': 'Residential buildings benefit from smart controls and scheduling',
                'action': 'Install smart plugs, Wi-Fi switches, and energy monitoring system for major appliances.',
                'cost': '₹15,000 - ₹40,000',
                'savings': '10-15% through better control and phantom load elimination',
                'payback': '2-3 years'
            })
        
        # Budget-based quick wins
        if profile['budget'] == 'a':
            recommendations['high_priority'].append({
                'title': 'Low-Cost Quick Wins Package',
                'reason': 'Maximum impact within your budget constraint',
                'action': 'Combo: LED bulbs + door sweeps + ceiling fans + power strips + window films for west/south windows.',
                'cost': '₹25,000 - ₹45,000',
                'savings': '15-20% total energy reduction',
                'payback': '18-24 months'
            })
        
        # High consumption alerts
        consumption_per_sqft = (avg_power * 24) / profile['floor_area']
        if consumption_per_sqft > 15:
            recommendations['high_priority'].append({
                'title': 'Energy Audit - High Consumption Detected',
                'reason': f'Your consumption of {consumption_per_sqft:.1f} Wh/sq.ft/day is above normal ({">20%" if consumption_per_sqft > 18 else "10-20%"})',
                'action': 'Schedule professional energy audit to identify specific inefficiencies and ghost loads.',
                'cost': '₹5,000 - ₹15,000',
                'savings': 'Audit will reveal 20-30% saving opportunities',
                'payback': 'Immediate through identified savings'
            })
        
        # Voltage stabilization
        if self.df is not None:
            voltage_std = self.df['Voltage'].std()
            if voltage_std > 8:
                recommendations['medium_priority'].append({
                    'title': 'Voltage Stabilizer Installation',
                    'reason': f'High voltage fluctuation detected (std dev: {voltage_std:.1f}V) - damaging equipment and wasting energy',
                    'action': 'Install whole-house/building voltage stabilizer or mainline stabilizer.',
                    'cost': '₹15,000 - ₹35,000',
                    'savings': '8-12% energy savings + equipment protection',
                    'payback': '18-24 months'
                })
        
        # Ensure at least one in each category
        if not recommendations['low_priority']:
            recommendations['low_priority'].append({
                'title': 'Energy Monitoring Dashboard',
                'reason': 'Track and optimize your energy usage patterns',
                'action': 'Install real-time energy monitoring system with mobile app.',
                'cost': '₹8,000 - ₹20,000',
                'savings': '5-10% through behavioral changes and insights',
                'payback': '2-3 years'
            })
        
        return recommendations

# Initialize analyzer
energy_analyzer = EnergyAnalysisSystem()

# API Routes
@app.route('/')
def index():
    return send_from_directory('.', 'dashboard.html')

@app.route('/api/refresh-data', methods=['GET'])
def refresh_data():
    global last_fetch_time
    try:
        df = energy_analyzer.fetch_data_from_sheets()
        if df is not None:
            last_fetch_time = datetime.now()
            return jsonify({
                'success': True,
                'message': 'Data refreshed successfully',
                'rows': len(df),
                'timestamp': datetime.now().isoformat()
            })
        return jsonify({'success': False, 'message': 'Failed to fetch data'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/latest-data', methods=['GET'])
def get_latest_data():
    global last_fetch_time
    
    if last_fetch_time is None or (datetime.now() - last_fetch_time).seconds > CACHE_DURATION:
        energy_analyzer.fetch_data_from_sheets()
        last_fetch_time = datetime.now()
    
    if energy_analyzer.df is None or len(energy_analyzer.df) == 0:
        return jsonify({'success': False, 'message': 'No data available'}), 404
    
    latest = energy_analyzer.df.iloc[-1]
    stats = energy_analyzer.monthly_energy_report()
    
    return jsonify({
        'success': True,
        'data': {
            'voltage': float(latest['Voltage']),
            'current': float(latest['Current']),
            'power': float(latest['Power']),
            'powerFactor': float(latest['PowerFactor']),
            'efficiency': float(latest['Efficiency']),
            'energy': stats['totalEnergy'] if stats else 0,
            'cost': stats['monthlyCost'] if stats else 0,
            'timestamp': latest['Timestamp'].isoformat()
        }
    })

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    report = energy_analyzer.monthly_energy_report()
    efficiency = energy_analyzer.batch_efficiency_analysis()
    
    if report and efficiency:
        return jsonify({
            'success': True,
            'data': {**report, **efficiency}
        })
    return jsonify({'success': False, 'message': 'No data available'}), 404

@app.route('/api/time-series', methods=['GET'])
def get_time_series():
    limit = request.args.get('limit', 30, type=int)
    
    if energy_analyzer.df is None:
        return jsonify({'success': False, 'message': 'No data available'}), 404
    
    df_subset = energy_analyzer.df.tail(limit)
    
    return jsonify({
        'success': True,
        'data': {
            'timestamps': df_subset['Timestamp'].astype(str).tolist(),
            'power': df_subset['Power'].tolist(),
            'voltage': df_subset['Voltage'].tolist(),
            'current': df_subset['Current'].tolist(),
            'efficiency': df_subset['Efficiency'].tolist()
        }
    })

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    hours = request.args.get('hours', 24, type=int)
    predictions = energy_analyzer.ai_energy_prediction(hours)
    
    if predictions:
        return jsonify({'success': True, 'data': predictions})
    return jsonify({'success': False, 'message': 'Unable to generate predictions'}), 500

@app.route('/api/peak-hours', methods=['GET'])
def get_peak_hours():
    peak_data = energy_analyzer.detect_peak_hours()
    
    if peak_data:
        return jsonify({'success': True, 'data': peak_data})
    return jsonify({'success': False, 'message': 'No data available'}), 404

@app.route('/api/carbon-stats', methods=['GET'])
def get_carbon_stats():
    carbon_data = energy_analyzer.carbon_credit_tracking()
    
    if carbon_data:
        return jsonify({'success': True, 'data': carbon_data})
    return jsonify({'success': False, 'message': 'No data available'}), 404

@app.route('/api/appliance-breakdown', methods=['GET'])
def get_appliance_breakdown():
    breakdown = energy_analyzer.appliance_breakdown()
    
    if breakdown:
        return jsonify({'success': True, 'data': breakdown})
    return jsonify({'success': False, 'message': 'No data available'}), 404

@app.route('/api/run-analysis', methods=['POST'])
def run_analysis():
    try:
        stats = energy_analyzer.monthly_energy_report()
        efficiency = energy_analyzer.batch_efficiency_analysis()
        peak_hours = energy_analyzer.detect_peak_hours()
        carbon = energy_analyzer.carbon_credit_tracking()
        appliances = energy_analyzer.appliance_breakdown()
        
        return jsonify({
            'success': True,
            'data': {
                'statistics': stats,
                'efficiency': efficiency,
                'peakHours': peak_hours,
                'carbon': carbon,
                'appliances': appliances
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get personalized recommendations based on user profile"""
    try:
        profile = request.json
        
        if energy_analyzer.df is None:
            return jsonify({'success': False, 'message': 'No data available'}), 404
        
        avg_power = float(energy_analyzer.df['Power'].mean())
        peak_power = float(energy_analyzer.df['Power'].max())
        current_pf = float(energy_analyzer.df['PowerFactor'].mean())
        
        recommendations = energy_analyzer.generate_personalized_recommendations(
            profile, avg_power, peak_power, current_pf
        )
        
        return jsonify({
            'success': True,
            'data': recommendations
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'dataPoints': len(energy_analyzer.df) if energy_analyzer.df is not None else 0,
        'lastFetch': last_fetch_time.isoformat() if last_fetch_time else None,
        'sheetId': SPREADSHEET_ID
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 COMPLETE ENERGY MONITORING DASHBOARD - STARTING UP")
    print("="*70)
    print(f"\n📊 Configuration:")
    print(f"   Google Sheet ID: {SPREADSHEET_ID}")
    print(f"   Sheet Name: {SHEET_NAME}")
    print(f"   Cache Duration: {CACHE_DURATION} seconds")
    print(f"\n✨ Features Loaded:")
    print(f"   ✓ AI Energy Predictions (24-hour forecast)")
    print(f"   ✓ Batch Efficiency Analysis")
    print(f"   ✓ Monthly Energy Reports")
    print(f"   ✓ Peak Hour Detection")
    print(f"   ✓ Appliance-Level Breakdown (ML Clustering)")
    print(f"   ✓ Carbon Credit Tracking")
    print(f"   ✓ Personalized Recommendations")
    print(f"   ✓ Real-time Google Sheets Integration")
    print(f"\n🔄 Fetching initial data from Google Sheets...")
    
    result = energy_analyzer.fetch_data_from_sheets()
    
    if result is not None:
        print(f"\n✅ SUCCESS! Dashboard is ready!")
        print(f"   Data Points Loaded: {len(result)}")
        print(f"   Time Range: {result['Timestamp'].min()} to {result['Timestamp'].max()}")
        print(f"   Average Power: {result['Power'].mean():.2f} W")
        print(f"   Peak Power: {result['Power'].max():.2f} W")
    else:
        print(f"\n⚠️  WARNING: Could not fetch initial data")
        print(f"   Dashboard will still start but data may be unavailable")
        print(f"   Please check:")
        print(f"   1. Google Sheet is public (Share > Anyone with link)")
        print(f"   2. Sheet ID is correct: {SPREADSHEET_ID}")
        print(f"   3. Sheet has columns: Voltage, Current, Power")
        print(f"   4. Internet connection is active")
    
    print(f"\n🌐 Starting Flask server...")
    print(f"   Local Access: http://localhost:5000")
    print(f"   Network Access: http://YOUR_IP:5000")
    print(f"\n📡 Auto-update: Data refreshes every {CACHE_DURATION} seconds")
    print(f"\n🎯 Available API Endpoints:")
    print(f"   GET  /api/latest-data        - Latest sensor reading")
    print(f"   GET  /api/statistics         - Overall statistics")
    print(f"   GET  /api/time-series        - Chart data")
    print(f"   GET  /api/predictions        - AI predictions")
    print(f"   GET  /api/peak-hours         - Peak hour analysis")
    print(f"   GET  /api/carbon-stats       - Carbon tracking")
    print(f"   GET  /api/appliance-breakdown - Appliance analysis")
    print(f"   POST /api/run-analysis       - Full analysis")
    print(f"   POST /api/recommendations    - Get recommendations")
    print(f"   GET  /api/health             - System health")
    print(f"\n💡 Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)