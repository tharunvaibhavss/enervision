import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

class EnergyAnalysisSystem:
    def __init__(self, data_file):
        """Initialize the energy analysis system"""
        try:
            # Read CSV without timestamp
            self.df = pd.read_csv(data_file)
            
            # Print available columns for debugging
            print("Available columns:", self.df.columns.tolist())
            
            # Create synthetic timestamp based on row index
            start_time = datetime(2025, 11, 18, 1, 0, 0)
            self.df['Timestamp'] = [start_time + timedelta(seconds=i*6) for i in range(len(self.df))]
            
            print(f"✓ Loaded {len(self.df)} data points successfully")
            print(f"✓ Time range: {self.df['Timestamp'].min()} to {self.df['Timestamp'].max()}")
            
            self.prepare_features()
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
        
    def prepare_features(self):
        """Extract time-based features for ML models"""
        self.df['Hour'] = self.df['Timestamp'].dt.hour
        self.df['DayOfWeek'] = self.df['Timestamp'].dt.dayofweek
        self.df['Month'] = self.df['Timestamp'].dt.month
        self.df['DayOfMonth'] = self.df['Timestamp'].dt.day
        self.df['IsWeekend'] = (self.df['DayOfWeek'] >= 5).astype(int)
        
        # Rolling averages for pattern detection
        self.df['Power_MA_5'] = self.df['Power'].rolling(window=5, min_periods=1).mean()
        self.df['Power_MA_10'] = self.df['Power'].rolling(window=10, min_periods=1).mean()
        self.df['Voltage_Stability'] = self.df['Voltage'].rolling(window=5, min_periods=1).std()
        
    def ai_energy_prediction(self, hours_ahead=24):
        """AI-powered energy prediction using ensemble methods"""
        print("\n" + "=" * 70)
        print("AI-POWERED ENERGY PREDICTION")
        print("=" * 70)
        
        # Prepare training data
        features = ['Hour', 'DayOfWeek', 'Month', 'Voltage', 'Current', 
                   'Power_MA_5', 'Power_MA_10', 'Voltage_Stability']
        X = self.df[features].fillna(method='bfill')
        y = self.df['Power']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Ensemble model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        rf_model.fit(X_train_scaled, y_train)
        gb_model.fit(X_train_scaled, y_train)
        
        # Predictions
        rf_pred = rf_model.predict(X_test_scaled)
        gb_pred = gb_model.predict(X_test_scaled)
        ensemble_pred = (rf_pred + gb_pred) / 2
        
        # Model accuracy
        rf_score = rf_model.score(X_test_scaled, y_test)
        gb_score = gb_model.score(X_test_scaled, y_test)
        
        print(f"\nModel Performance:")
        print(f"  Random Forest R² Score: {rf_score:.4f}")
        print(f"  Gradient Boosting R² Score: {gb_score:.4f}")
        print(f"  Ensemble Accuracy: {((rf_score + gb_score) / 2):.4f}")
        
        # Visualization 1: Actual vs Predicted
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Actual vs Predicted Scatter
        axes[0, 0].scatter(y_test, ensemble_pred, alpha=0.5, s=20)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Power (W)')
        axes[0, 0].set_ylabel('Predicted Power (W)')
        axes[0, 0].set_title('AI Prediction Accuracy: Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Model Comparison
        models = ['Random Forest', 'Gradient Boost', 'Ensemble']
        scores = [rf_score, gb_score, (rf_score + gb_score)/2]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        axes[0, 1].bar(models, scores, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_title('Model Performance Comparison')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(scores):
            axes[0, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
        
        # Plot 3: Feature Importance
        feature_importance = rf_model.feature_importances_
        feature_names = features
        indices = np.argsort(feature_importance)[::-1]
        axes[1, 0].barh(range(len(indices)), feature_importance[indices], color='#9b59b6', alpha=0.7)
        axes[1, 0].set_yticks(range(len(indices)))
        axes[1, 0].set_yticklabels([feature_names[i] for i in indices])
        axes[1, 0].set_xlabel('Importance Score')
        axes[1, 0].set_title('Feature Importance Analysis')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Future Predictions
        last_timestamp = self.df['Timestamp'].max()
        future_timestamps = [last_timestamp + timedelta(hours=i) for i in range(1, hours_ahead + 1)]
        future_predictions = []
        
        for ts in future_timestamps:
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
            future_predictions.append((rf_p + gb_p) / 2)
        
        axes[1, 1].plot(range(hours_ahead), future_predictions, marker='o', linewidth=2, 
                       markersize=6, color='#e74c3c', label='Predicted')
        axes[1, 1].fill_between(range(hours_ahead), 
                               np.array(future_predictions) * 0.95, 
                               np.array(future_predictions) * 1.05, 
                               alpha=0.2, color='#e74c3c')
        axes[1, 1].set_xlabel('Hours Ahead')
        axes[1, 1].set_ylabel('Predicted Power (W)')
        axes[1, 1].set_title('24-Hour Energy Forecast')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('01_ai_prediction.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved: 01_ai_prediction.png")
        plt.show()
        
        # Print forecast table
        print(f"\n24-Hour Energy Forecast:")
        print(f"{'Time':<20} {'Predicted Power (W)':<20} {'Confidence'}")
        print("-" * 70)
        
        for i, (ts, pred) in enumerate(zip(future_timestamps[:12], future_predictions[:12])):
            confidence = min(rf_score, gb_score) * 100
            print(f"{ts.strftime('%Y-%m-%d %H:%M'):<20} {pred:<20.2f} {confidence:.1f}%")
        
        print(f"\n... and {hours_ahead - 12} more hours")
        
        return rf_model, gb_model, scaler
    
    def batch_efficiency_analysis(self):
        """Batch processing for efficiency patterns"""
        print("\n" + "=" * 70)
        print("BATCH PROCESSING & EFFICIENCY ANALYSIS")
        print("=" * 70)
        
        # Power factor analysis
        self.df['PowerFactor'] = self.df['Power'] / (self.df['Voltage'] * self.df['Current'])
        self.df['Efficiency'] = self.df['PowerFactor'] * 100
        
        print(f"\nSystem Efficiency Metrics:")
        print(f"  Average Power Factor: {self.df['PowerFactor'].mean():.4f}")
        print(f"  Average Efficiency: {self.df['Efficiency'].mean():.2f}%")
        print(f"  Peak Efficiency: {self.df['Efficiency'].max():.2f}%")
        print(f"  Minimum Efficiency: {self.df['Efficiency'].min():.2f}%")
        
        # Voltage stability
        voltage_std = self.df['Voltage'].std()
        
        print(f"\nVoltage Stability Analysis:")
        print(f"  Voltage Range: {self.df['Voltage'].min():.2f}V - {self.df['Voltage'].max():.2f}V")
        print(f"  Standard Deviation: {voltage_std:.2f}V")
        print(f"  Stability Rating: {'Excellent' if voltage_std < 5 else 'Good' if voltage_std < 10 else 'Fair'}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Efficiency Over Time
        axes[0, 0].plot(self.df.index, self.df['Efficiency'], color='#2ecc71', alpha=0.6, linewidth=1)
        axes[0, 0].axhline(y=self.df['Efficiency'].mean(), color='r', linestyle='--', 
                          label=f'Mean: {self.df["Efficiency"].mean():.2f}%')
        axes[0, 0].fill_between(self.df.index, 95, 100, alpha=0.2, color='green', label='Optimal Range')
        axes[0, 0].set_xlabel('Reading Number')
        axes[0, 0].set_ylabel('Efficiency (%)')
        axes[0, 0].set_title('System Efficiency Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Power Factor Distribution
        axes[0, 1].hist(self.df['PowerFactor'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(self.df['PowerFactor'].mean(), color='r', linestyle='--', 
                          linewidth=2, label=f'Mean: {self.df["PowerFactor"].mean():.3f}')
        axes[0, 1].set_xlabel('Power Factor')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Power Factor Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Voltage Stability
        axes[1, 0].plot(self.df.index, self.df['Voltage'], color='#e74c3c', alpha=0.7, linewidth=1)
        axes[1, 0].fill_between(self.df.index, 230, 250, alpha=0.2, color='green', label='Safe Range (230-250V)')
        axes[1, 0].axhline(y=self.df['Voltage'].mean(), color='blue', linestyle='--', 
                          label=f'Mean: {self.df["Voltage"].mean():.2f}V')
        axes[1, 0].set_xlabel('Reading Number')
        axes[1, 0].set_ylabel('Voltage (V)')
        axes[1, 0].set_title('Voltage Stability Analysis')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Efficiency vs Power Factor
        scatter = axes[1, 1].scatter(self.df['PowerFactor'], self.df['Efficiency'], 
                                    c=self.df['Power'], cmap='viridis', alpha=0.6, s=30)
        axes[1, 1].set_xlabel('Power Factor')
        axes[1, 1].set_ylabel('Efficiency (%)')
        axes[1, 1].set_title('Efficiency vs Power Factor (colored by Power)')
        plt.colorbar(scatter, ax=axes[1, 1], label='Power (W)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('02_efficiency_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved: 02_efficiency_analysis.png")
        plt.show()
        
        # Energy waste detection
        low_efficiency = self.df[self.df['Efficiency'] < 95]
        energy_waste = low_efficiency['Power'].sum() * 0.05 / 60  # Convert to Wh
        
        print(f"\nEnergy Waste Detection:")
        print(f"  Low Efficiency Periods: {len(low_efficiency)} readings ({len(low_efficiency)/len(self.df)*100:.1f}%)")
        print(f"  Estimated Energy Waste: {energy_waste:.2f} Wh")
        print(f"  Potential Savings: ₹{energy_waste * 0.008:.2f}/period (@ ₹8/kWh)")
        
    def monthly_energy_report(self):
        """Generate comprehensive monthly energy report"""
        print("\n" + "=" * 70)
        print("MONTHLY ENERGY REPORT")
        print("=" * 70)
        
        # Total consumption
        total_energy = self.df['Power'].sum() / 60  # Convert to Wh
        total_kwh = total_energy / 1000
        
        print(f"\nEnergy Consumption Summary:")
        print(f"  Total Energy Consumed: {total_kwh:.2f} kWh")
        print(f"  Average Power Draw: {self.df['Power'].mean():.2f} W")
        print(f"  Peak Power: {self.df['Power'].max():.2f} W")
        print(f"  Minimum Power: {self.df['Power'].min():.2f} W")
        
        # Cost analysis
        cost_per_kwh = 8  # ₹8 per kWh
        monthly_cost = total_kwh * cost_per_kwh
        
        print(f"\nCost Analysis:")
        print(f"  Estimated Cost: ₹{monthly_cost:.2f}")
        print(f"  Daily Average: ₹{monthly_cost / 30:.2f}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Power Consumption Over Time
        axes[0, 0].plot(self.df['Timestamp'], self.df['Power'], color='#3498db', linewidth=1)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Power (W)')
        axes[0, 0].set_title('Power Consumption Timeline')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Voltage, Current, Power Trends
        ax2 = axes[0, 1]
        ax2.plot(self.df.index, self.df['Voltage'], label='Voltage (V)', color='#e74c3c', alpha=0.7)
        ax2.set_xlabel('Reading Number')
        ax2.set_ylabel('Voltage (V)', color='#e74c3c')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(self.df.index, self.df['Current'], label='Current (A)', color='#2ecc71', alpha=0.7)
        ax2_twin.set_ylabel('Current (A)', color='#2ecc71')
        ax2_twin.tick_params(axis='y', labelcolor='#2ecc71')
        
        axes[0, 1].set_title('Voltage and Current Trends')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Hourly Distribution
        hourly_stats = self.df.groupby('Hour')['Power'].agg(['mean', 'max', 'min'])
        if len(hourly_stats) > 0:
            x_pos = hourly_stats.index
            axes[1, 0].bar(x_pos, hourly_stats['mean'], alpha=0.7, color='#3498db', label='Average')
            axes[1, 0].plot(x_pos, hourly_stats['max'], marker='o', color='#e74c3c', 
                           linewidth=2, markersize=4, label='Peak')
            axes[1, 0].set_xlabel('Hour of Day')
            axes[1, 0].set_ylabel('Power (W)')
            axes[1, 0].set_title('Hourly Power Consumption Pattern')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Power Distribution
        axes[1, 1].hist(self.df['Power'], bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(self.df['Power'].mean(), color='r', linestyle='--', 
                          linewidth=2, label=f'Mean: {self.df["Power"].mean():.0f}W')
        axes[1, 1].axvline(self.df['Power'].median(), color='g', linestyle='--', 
                          linewidth=2, label=f'Median: {self.df["Power"].median():.0f}W')
        axes[1, 1].set_xlabel('Power (W)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Power Consumption Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('03_monthly_report.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved: 03_monthly_report.png")
        plt.show()
        
    def detect_peak_hours(self):
        """Advanced peak hour detection with anomaly identification"""
        print("\n" + "=" * 70)
        print("PEAK HOUR DETECTION & LOAD ANALYSIS")
        print("=" * 70)
        
        hourly_avg = self.df.groupby('Hour')['Power'].mean()
        
        # Check if we have enough data
        if len(hourly_avg) == 0 or hourly_avg.isna().all():
            print("\n⚠️ Warning: Insufficient data for peak hour analysis")
            return
        
        # Use a more lenient threshold if we have limited hours
        if len(hourly_avg) < 5:
            threshold = hourly_avg.mean() + hourly_avg.std() * 0.5
        else:
            threshold = hourly_avg.mean() + hourly_avg.std()
        
        peak_hours = hourly_avg[hourly_avg > threshold]
        
        # If no peak hours found, use top 3 hours
        if len(peak_hours) == 0:
            peak_hours = hourly_avg.nlargest(min(3, len(hourly_avg)))
            print(f"\nTop Consumption Hours (using top {len(peak_hours)} hours):")
        else:
            print(f"\nPeak Hours Identified (>{threshold:.2f}W):")
        
        print(f"{'Hour':<15} {'Average Power (W)':<20} {'% Above Baseline'}")
        print("-" * 70)
        
        for hour, power in peak_hours.items():
            pct_above = ((power - hourly_avg.mean()) / hourly_avg.mean()) * 100
            print(f"{hour:02d}:00 - {hour+1:02d}:00    {power:<20.2f} +{pct_above:.1f}%")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Hourly Load Profile
        hours = hourly_avg.index
        colors = ['#e74c3c' if hour in peak_hours.index else '#3498db' for hour in hours]
        axes[0, 0].bar(hours, hourly_avg.values, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 0].axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                          label=f'Peak Threshold: {threshold:.0f}W')
        axes[0, 0].axhline(y=hourly_avg.mean(), color='green', linestyle='--', linewidth=2,
                          label=f'Average: {hourly_avg.mean():.0f}W')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Average Power (W)')
        axes[0, 0].set_title('24-Hour Load Profile (Red = Peak Hours)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        if len(hourly_avg) > 12:
            axes[0, 0].set_xticks(range(0, 24, 2))
        
        # Plot 2: Load Duration Curve
        sorted_power = np.sort(self.df['Power'].values)[::-1]
        percentiles = np.linspace(0, 100, len(sorted_power))
        axes[0, 1].plot(percentiles, sorted_power, color='#2ecc71', linewidth=2)
        axes[0, 1].fill_between(percentiles, sorted_power, alpha=0.3, color='#2ecc71')
        axes[0, 1].set_xlabel('Percentage of Time (%)')
        axes[0, 1].set_ylabel('Power (W)')
        axes[0, 1].set_title('Load Duration Curve')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=hourly_avg.mean(), color='red', linestyle='--', 
                          label=f'Average Load: {hourly_avg.mean():.0f}W')
        axes[0, 1].legend()
        
        # Plot 3: Peak vs Off-Peak Distribution (Box Plot instead of Violin)
        peak_hour_data = self.df[self.df['Hour'].isin(peak_hours.index)]['Power']
        offpeak_hour_data = self.df[~self.df['Hour'].isin(peak_hours.index)]['Power']
        
        # Only plot if we have data in both categories
        if len(peak_hour_data) > 0 and len(offpeak_hour_data) > 0:
            data_to_plot = [peak_hour_data, offpeak_hour_data]
            bp = axes[1, 0].boxplot(data_to_plot, labels=['Peak Hours', 'Off-Peak Hours'],
                                    patch_artist=True, showmeans=True)
            for patch, color in zip(bp['boxes'], ['#e74c3c', '#3498db']):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            axes[1, 0].set_ylabel('Power (W)')
            axes[1, 0].set_title('Peak vs Off-Peak Power Distribution')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient data for comparison', 
                           ha='center', va='center', transform=axes[1, 0].transAxes,
                           fontsize=12)
            axes[1, 0].set_title('Peak vs Off-Peak Power Distribution')
        
        # Plot 4: Time-of-Use Cost Analysis
        if len(peak_hour_data) > 0 and len(offpeak_hour_data) > 0:
            peak_cost = sum(peak_hour_data) / 60000 * 12  # ₹12/kWh peak rate
            offpeak_cost = sum(offpeak_hour_data) / 60000 * 6  # ₹6/kWh off-peak rate
        else:
            # Use total consumption with estimated split
            total_consumption = sum(self.df['Power']) / 60000
            peak_cost = total_consumption * 0.4 * 12
            offpeak_cost = total_consumption * 0.6 * 6
        
        categories = ['Current\nCost', 'With Peak\nRates', 'With Load\nShifting']
        costs = [
            (sum(self.df['Power']) / 60000) * 8,  # Current flat rate
            peak_cost + offpeak_cost,  # TOU rates
            peak_cost * 0.7 + offpeak_cost * 1.1  # After load shifting
        ]
        colors_cost = ['#95a5a6', '#e74c3c', '#2ecc71']
        
        bars = axes[1, 1].bar(categories, costs, color=colors_cost, alpha=0.7, edgecolor='black')
        axes[1, 1].set_ylabel('Cost (₹)')
        axes[1, 1].set_title('Cost Impact: Time-of-Use Analysis')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        for i, (bar, cost) in enumerate(zip(bars, costs)):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'₹{cost:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('04_peak_hours.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved: 04_peak_hours.png")
        plt.show()
        
        print(f"\nLoad Profile Analysis:")
        print(f"  Base Load: {hourly_avg.min():.2f} W")
        print(f"  Average Load: {hourly_avg.mean():.2f} W")
        print(f"  Peak Load: {hourly_avg.max():.2f} W")
        print(f"  Load Factor: {(hourly_avg.mean() / hourly_avg.max()):.2%}")
        
        if len(peak_hour_data) > 0 and len(offpeak_hour_data) > 0:
            print(f"\nTime-of-Use Cost Optimization:")
            print(f"  Peak Hour Cost: ₹{peak_cost:.2f}")
            print(f"  Off-Peak Hour Cost: ₹{offpeak_cost:.2f}")
            print(f"  Potential Savings with Load Shifting: ₹{peak_cost * 0.3:.2f}")
        
    def appliance_breakdown(self):
        """Appliance-level consumption estimation using ML clustering"""
        print("\n" + "=" * 70)
        print("APPLIANCE-LEVEL CONSUMPTION BREAKDOWN")
        print("=" * 70)
        
        # K-means clustering for appliance identification
        X_cluster = self.df[['Power', 'Current', 'Voltage']].values
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['Cluster'] = kmeans.fit_predict(X_cluster)
        
        # Estimate appliance signatures
        power_ranges = {
            'Base Load (Standby)': (0, 100),
            'Lighting': (100, 300),
            'Fans/Small Appliances': (300, 600),
            'Computer/TV': (600, 800),
            'Refrigerator/AC': (800, 1000),
            'High Power Equipment': (1000, 1500)
        }
        
        print(f"\nEstimated Appliance Consumption:")
        print(f"{'Appliance Category':<30} {'Avg Power (W)':<15} {'Energy (Wh)':<15} {'% of Total'}")
        print("-" * 70)
        
        total_power = self.df['Power'].sum()
        appliance_data = []
        
        for category, (min_pow, max_pow) in power_ranges.items():
            subset = self.df[(self.df['Power'] >= min_pow) & (self.df['Power'] < max_pow)]
            if len(subset) > 0:
                avg_power = subset['Power'].mean()
                energy_wh = subset['Power'].sum() / 60
                pct_total = (subset['Power'].sum() / total_power) * 100
                appliance_data.append({
                    'category': category,
                    'avg_power': avg_power,
                    'energy': energy_wh,
                    'pct': pct_total
                })
                print(f"{category:<30} {avg_power:<15.2f} {energy_wh:<15.2f} {pct_total:.1f}%")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Appliance Breakdown Pie Chart
        if appliance_data:
            categories = [d['category'].split('(')[0].strip() for d in appliance_data]
            percentages = [d['pct'] for d in appliance_data]
            colors_pie = plt.cm.Set3(range(len(categories)))
            
            wedges, texts, autotexts = axes[0, 0].pie(percentages, labels=categories, autopct='%1.1f%%',
                                                       colors=colors_pie, startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            axes[0, 0].set_title('Energy Consumption by Appliance Category')
        
        # Plot 2: Power Consumption Bar Chart
        if appliance_data:
            cat_names = [d['category'].split('(')[0].strip()[:15] for d in appliance_data]
            avg_powers = [d['avg_power'] for d in appliance_data]
            colors_bar = plt.cm.viridis(np.linspace(0, 1, len(cat_names)))
            
            bars = axes[0, 1].barh(cat_names, avg_powers, color=colors_bar, alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Average Power (W)')
            axes[0, 1].set_title('Average Power by Appliance Category')
            axes[0, 1].grid(True, alpha=0.3, axis='x')
            
            for bar, power in zip(bars, avg_powers):
                width = bar.get_width()
                axes[0, 1].text(width, bar.get_y() + bar.get_height()/2., 
                               f'{power:.0f}W', ha='left', va='center', fontweight='bold')
        
        # Plot 3: K-Means Clustering Visualization
        scatter = axes[1, 0].scatter(self.df['Current'], self.df['Power'], 
                                     c=self.df['Cluster'], cmap='tab10', alpha=0.6, s=30)
        centers = kmeans.cluster_centers_
        axes[1, 0].scatter(centers[:, 1], centers[:, 0], c='red', s=200, alpha=0.8, 
                          marker='X', edgecolor='black', linewidth=2, label='Cluster Centers')
        axes[1, 0].set_xlabel('Current (A)')
        axes[1, 0].set_ylabel('Power (W)')
        axes[1, 0].set_title('ML-Based Appliance Clustering')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='Cluster ID')
        
        # Plot 4: Power Heatmap by Hour
        if len(self.df['Hour'].unique()) > 1:
            pivot_data = self.df.pivot_table(values='Power', index='Hour', 
                                            columns=self.df.index // 10, aggfunc='mean')
            sns.heatmap(pivot_data, cmap='YlOrRd', ax=axes[1, 1], cbar_kws={'label': 'Power (W)'})
            axes[1, 1].set_xlabel('Time Period')
            axes[1, 1].set_ylabel('Hour of Day')
            axes[1, 1].set_title('Power Consumption Heatmap')
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient hourly data for heatmap', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Power Consumption Heatmap')
        
        plt.tight_layout()
        plt.savefig('05_appliance_breakdown.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved: 05_appliance_breakdown.png")
        plt.show()
        
        print(f"\nTop Energy Consumers (Recommendations):")
        print("  1. AC/Refrigerator: Consider energy-efficient models (5-star rated)")
        print("  2. Lighting: Switch to LED bulbs (80% energy savings)")
        print("  3. Standby Power: Use smart plugs to eliminate phantom loads")
        
    def carbon_credit_tracking(self):
        """Carbon emissions and credit generation tracking"""
        print("\n" + "=" * 70)
        print("CARBON CREDIT GENERATION & TRACKING")
        print("=" * 70)
        
        # Calculate carbon emissions
        total_kwh = self.df['Power'].sum() / 60000
        carbon_intensity = 0.82  # kg CO2 per kWh (India grid average)
        total_emissions = total_kwh * carbon_intensity
        
        print(f"\nCarbon Footprint Analysis:")
        print(f"  Total Energy Consumed: {total_kwh:.2f} kWh")
        print(f"  Carbon Emissions: {total_emissions:.2f} kg CO2")
        print(f"  Monthly Projection: {total_emissions * 30:.2f} kg CO2")
        print(f"  Annual Projection: {total_emissions * 365:.2f} kg CO2 ({total_emissions * 365 / 1000:.2f} tonnes)")
        
        # Carbon credit potential
        efficiency_improvement = 0.15  # 15% improvement potential
        carbon_saved = total_emissions * efficiency_improvement
        carbon_credits = carbon_saved / 1000  # 1 credit per tonne
        credit_value = carbon_credits * 1200  # ₹1200 per credit
        
        print(f"\nCarbon Credit Potential:")
        print(f"  Achievable CO2 Reduction: {carbon_saved:.2f} kg")
        print(f"  Carbon Credits (Monthly): {carbon_credits:.4f} credits")
        print(f"  Credit Value: ₹{credit_value:.2f}/month")
        print(f"  Annual Credit Value: ₹{credit_value * 12:.2f}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Emissions Timeline
        cumulative_emissions = (self.df['Power'].cumsum() / 60000) * carbon_intensity
        axes[0, 0].plot(self.df.index, cumulative_emissions, color='#e74c3c', linewidth=2)
        axes[0, 0].fill_between(self.df.index, cumulative_emissions, alpha=0.3, color='#e74c3c')
        axes[0, 0].set_xlabel('Reading Number')
        axes[0, 0].set_ylabel('Cumulative CO2 (kg)')
        axes[0, 0].set_title('Cumulative Carbon Emissions')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Carbon Reduction Potential
        scenarios = ['Current', 'With 15%\nReduction', 'With 30%\nReduction', 'With Solar\n(95%)']
        emissions = [
            total_emissions,
            total_emissions * 0.85,
            total_emissions * 0.70,
            total_emissions * 0.05
        ]
        colors_scenarios = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71']
        
        bars = axes[0, 1].bar(scenarios, emissions, color=colors_scenarios, alpha=0.7, edgecolor='black')
        axes[0, 1].set_ylabel('CO2 Emissions (kg)')
        axes[0, 1].set_title('Carbon Reduction Scenarios')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        for bar, emission in zip(bars, emissions):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{emission:.1f}kg', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Carbon Credit Value Over Time
        months = ['Month 1', 'Month 6', 'Month 12', 'Year 2', 'Year 3']
        credit_values = [credit_value * i for i in [1, 6, 12, 24, 36]]
        
        axes[1, 0].plot(months, credit_values, marker='o', linewidth=2, markersize=10,
                       color='#2ecc71')
        axes[1, 0].fill_between(range(len(months)), credit_values, alpha=0.3, color='#2ecc71')
        axes[1, 0].set_ylabel('Cumulative Credit Value (₹)')
        axes[1, 0].set_title('Carbon Credit Revenue Projection')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        for i, val in enumerate(credit_values):
            axes[1, 0].text(i, val, f'₹{val:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Renewable Energy Impact
        solar_offset = total_kwh * 0.95
        trees_equivalent = total_emissions / 21
        
        impact_categories = ['Trees\nEquivalent', 'Solar kW\nNeeded', 'CO2 Saved\n(kg)']
        impact_values = [trees_equivalent, solar_offset / 4, carbon_saved]
        colors_impact = ['#27ae60', '#f39c12', '#3498db']
        
        bars = axes[1, 1].bar(impact_categories, impact_values, color=colors_impact, 
                             alpha=0.7, edgecolor='black')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Environmental Impact Metrics')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, impact_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('06_carbon_tracking.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved: 06_carbon_tracking.png")
        plt.show()
        
        print(f"\nRenewable Energy Impact:")
        print(f"  Solar Offset Needed: {solar_offset:.2f} kWh")
        print(f"  Equivalent Trees Required: {trees_equivalent:.1f} trees")
        print(f"  Rooftop Solar Capacity: {solar_offset / 4:.1f} kW system")
        
    def architectural_recommendations(self):
        print("\n" + "=" * 70)
        print("ARCHITECTURAL DESIGN RECOMMENDATIONS")
        print("=" * 70)
        
        avg_power = self.df['Power'].mean()
        peak_power = self.df['Power'].max()
        avg_current = self.df['Current'].mean()
        current_pf = (self.df['Power'] / (self.df['Voltage'] * self.df['Current'])).mean()
        
        print(f"\nSystem Analysis:")
        print(f"  Current Load: {avg_power:.2f} W average, {peak_power:.2f} W peak")
        print(f"  Current Draw: {avg_current:.2f} A average")
        print(f"  Power Factor: {current_pf:.3f}")
        
        # Interactive questionnaire for personalized recommendations
        print("\n" + "=" * 70)
        print("📋 PERSONALIZED ASSESSMENT QUESTIONNAIRE")
        print("=" * 70)
        print("\nPlease answer the following questions for customized recommendations:")
        print("(Press Enter to skip any question)\n")
        
        # Collect user inputs
        user_profile = {}
        
        try:
            # Building Type
            print("1. What type of building is this?")
            print("   a) Residential home")
            print("   b) Commercial office")
            print("   c) Industrial facility")
            print("   d) Mixed-use building")
            building_type = input("   Your answer (a/b/c/d): ").strip().lower() or 'a'
            user_profile['building_type'] = building_type
            
            # Building Age
            print("\n2. How old is your building?")
            print("   a) Less than 5 years")
            print("   b) 5-15 years")
            print("   c) 15-30 years")
            print("   d) More than 30 years")
            building_age = input("   Your answer (a/b/c/d): ").strip().lower() or 'b'
            user_profile['building_age'] = building_age
            
            # Floor Area
            print("\n3. What is the approximate floor area? (in sq.ft)")
            area_input = input("   Your answer: ").strip()
            floor_area = float(area_input) if area_input else 1500
            user_profile['floor_area'] = floor_area
            
            # Current AC/Cooling
            print("\n4. What cooling system do you currently use?")
            print("   a) Central AC")
            print("   b) Split AC units")
            print("   c) Window AC units")
            print("   d) Fans only")
            print("   e) No cooling system")
            cooling_system = input("   Your answer (a/b/c/d/e): ").strip().lower() or 'b'
            user_profile['cooling_system'] = cooling_system
            
            # Insulation
            print("\n5. Does your building have proper insulation?")
            print("   a) Yes, well insulated")
            print("   b) Partial insulation")
            print("   c) No insulation")
            insulation = input("   Your answer (a/b/c): ").strip().lower() or 'b'
            user_profile['insulation'] = insulation
            
            # Windows
            print("\n6. What type of windows do you have?")
            print("   a) Double-glazed/energy-efficient")
            print("   b) Single-pane glass")
            print("   c) Old wooden/metal frame")
            windows = input("   Your answer (a/b/c): ").strip().lower() or 'b'
            user_profile['windows'] = windows
            
            # Lighting
            print("\n7. What percentage of your lighting is LED?")
            print("   a) 80-100%")
            print("   b) 40-80%")
            print("   c) Less than 40%")
            lighting = input("   Your answer (a/b/c): ").strip().lower() or 'c'
            user_profile['lighting'] = lighting
            
            # Solar Interest
            print("\n8. Are you interested in solar panel installation?")
            print("   a) Yes, very interested")
            print("   b) Maybe, want to know more")
            print("   c) Not interested")
            solar_interest = input("   Your answer (a/b/c): ").strip().lower() or 'b'
            user_profile['solar_interest'] = solar_interest
            
            # Budget
            print("\n9. What is your budget range for energy upgrades?")
            print("   a) Under ₹50,000")
            print("   b) ₹50,000 - ₹2,00,000")
            print("   c) ₹2,00,000 - ₹5,00,000")
            print("   d) Above ₹5,00,000")
            budget = input("   Your answer (a/b/c/d): ").strip().lower() or 'b'
            user_profile['budget'] = budget
            
            # Occupancy
            print("\n10. How many hours per day is the building typically occupied?")
            occupancy_input = input("    Your answer (0-24): ").strip()
            occupancy_hours = float(occupancy_input) if occupancy_input else 12
            user_profile['occupancy_hours'] = occupancy_hours
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Questionnaire interrupted. Using default recommendations...\n")
            user_profile = {
                'building_type': 'a',
                'building_age': 'b',
                'floor_area': 1500,
                'cooling_system': 'b',
                'insulation': 'b',
                'windows': 'b',
                'lighting': 'c',
                'solar_interest': 'b',
                'budget': 'b',
                'occupancy_hours': 12
            }
        except Exception as e:
            print(f"\n⚠️ Error in questionnaire: {e}. Using default recommendations...\n")
            user_profile = {
                'building_type': 'a',
                'building_age': 'b',
                'floor_area': 1500,
                'cooling_system': 'b',
                'insulation': 'b',
                'windows': 'b',
                'lighting': 'c',
                'solar_interest': 'b',
                'budget': 'b',
                'occupancy_hours': 12
            }
        
        # Generate personalized recommendations
        print("\n" + "=" * 70)
        print("🎯 PERSONALIZED RECOMMENDATIONS BASED ON YOUR PROFILE")
        print("=" * 70)
        
        recommendations = self.generate_personalized_recommendations(user_profile, avg_power, peak_power, current_pf)
        
        # Priority recommendations
        print("\n🔴 HIGH PRIORITY (Immediate Action Required):")
        for i, rec in enumerate(recommendations['high_priority'], 1):
            print(f"\n  {i}. {rec['title']}")
            print(f"     • Why: {rec['reason']}")
            print(f"     • Action: {rec['action']}")
            print(f"     • Investment: {rec['cost']}")
            print(f"     • Expected Savings: {rec['savings']}")
            print(f"     • Payback: {rec['payback']}")
        
        print("\n🟡 MEDIUM PRIORITY (Plan within 6 months):")
        for i, rec in enumerate(recommendations['medium_priority'], 1):
            print(f"\n  {i}. {rec['title']}")
            print(f"     • Why: {rec['reason']}")
            print(f"     • Action: {rec['action']}")
            print(f"     • Investment: {rec['cost']}")
            print(f"     • Expected Savings: {rec['savings']}")
            print(f"     • Payback: {rec['payback']}")
        
        print("\n🟢 LOW PRIORITY (Future consideration):")
        for i, rec in enumerate(recommendations['low_priority'], 1):
            print(f"\n  {i}. {rec['title']}")
            print(f"     • Why: {rec['reason']}")
            print(f"     • Action: {rec['action']}")
            print(f"     • Investment: {rec['cost']}")
            print(f"     • Expected Savings: {rec['savings']}")
            print(f"     • Payback: {rec['payback']}")
        
        # Standard recommendations (keeping original content)
        print("\n" + "=" * 70)
        print("📚 STANDARD TECHNICAL RECOMMENDATIONS")
        print("=" * 70)
        
        print(f"\n🏗️ IMMEDIATE INFRASTRUCTURE UPGRADES:")
        print(f"  1. Voltage Stabilization:")
        print(f"     - Install automatic voltage regulator (AVR)")
        print(f"     - Expected savings: 8-12%")
        print(f"     - ROI: 18-24 months")
        
        print(f"\n  2. Power Factor Correction:")
        print(f"     - Current Power Factor: {current_pf:.3f}")
        print(f"     - Install capacitor banks for correction to 0.95+")
        print(f"     - Potential current reduction: {(1 - current_pf/0.95) * 100:.1f}%")
        
        print(f"\n  3. Load Balancing:")
        print(f"     - Implement 3-phase load distribution")
        print(f"     - Install phase monitors and auto-switching")
        print(f"     - Reduce neutral current losses: 5-8%")
        
        print(f"\n⚡ SMART AUTOMATION SYSTEMS:")
        print(f"  1. IoT-Based Energy Management:")
        print(f"     - Smart meters with real-time monitoring")
        print(f"     - Automated load shedding during peak hours")
        print(f"     - Occupancy-based lighting control")
        print(f"     - Expected savings: 20-30%")
        
        print(f"\n  2. HVAC Optimization:")
        print(f"     - Variable Frequency Drives (VFD) for motors")
        print(f"     - Smart thermostats with learning algorithms")
        print(f"     - Zone-based temperature control")
        print(f"     - Expected savings: 25-40% on cooling")
        
        print(f"\n🌞 RENEWABLE ENERGY INTEGRATION:")
        recommended_solar = (avg_power * 24) / 4000  # 4 sun hours average
        print(f"  1. Solar PV System:")
        print(f"     - Recommended Capacity: {recommended_solar:.1f} kW")
        print(f"     - Annual Generation: {recommended_solar * 1500:.0f} kWh")
        print(f"     - Investment: ₹{recommended_solar * 50000:.0f}")
        print(f"     - Payback Period: 4-5 years")
        
        print(f"\n  2. Energy Storage:")
        print(f"     - Battery backup: {peak_power * 4 / 1000:.1f} kWh")
        print(f"     - Enable peak shaving and load shifting")
        print(f"     - Grid independence: 60-80%")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Savings Potential
        upgrades = ['Voltage\nStabilizer', 'Power Factor\nCorrection', 'LED\nConversion', 
                   'Smart\nAutomation', 'Solar PV']
        savings_pct = [10, 8, 15, 25, 70]
        colors_savings = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(upgrades)))
        
        bars = axes[0, 0].barh(upgrades, savings_pct, color=colors_savings, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Energy Savings (%)')
        axes[0, 0].set_title('Potential Savings by Upgrade Type')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        for bar, saving in zip(bars, savings_pct):
            width = bar.get_width()
            axes[0, 0].text(width, bar.get_y() + bar.get_height()/2., 
                           f'{saving}%', ha='left', va='center', fontweight='bold')
        
        # Plot 2: ROI Timeline
        upgrades_roi = ['AVR', 'Capacitor\nBanks', 'LED', 'IoT System', 'Solar PV']
        roi_months = [21, 18, 12, 36, 54]
        investment_costs = [15000, 25000, 30000, 150000, recommended_solar * 50000]
        
        color_map = ['#e74c3c' if m > 36 else '#f39c12' if m > 24 else '#2ecc71' for m in roi_months]
        bars = axes[0, 1].bar(upgrades_roi, roi_months, color=color_map, alpha=0.7, edgecolor='black')
        axes[0, 1].set_ylabel('Payback Period (Months)')
        axes[0, 1].set_title('Return on Investment Timeline')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        for bar, months in zip(bars, roi_months):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{months}m', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Cost-Benefit Analysis
        years = np.arange(0, 6)
        cumulative_savings = []
        total_investment = sum(investment_costs)
        annual_savings = (avg_power * 24 * 365 / 1000) * 8 * 0.35  # 35% reduction
        
        for year in years:
            if year == 0:
                cumulative_savings.append(-total_investment)
            else:
                cumulative_savings.append(cumulative_savings[-1] + annual_savings)
        
        axes[1, 0].plot(years, cumulative_savings, marker='o', linewidth=2, markersize=8,
                       color='#3498db')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        axes[1, 0].fill_between(years, 0, cumulative_savings, where=np.array(cumulative_savings) > 0,
                               alpha=0.3, color='green', label='Profit')
        axes[1, 0].fill_between(years, 0, cumulative_savings, where=np.array(cumulative_savings) < 0,
                               alpha=0.3, color='red', label='Investment')
        axes[1, 0].set_xlabel('Years')
        axes[1, 0].set_ylabel('Cumulative Value (₹)')
        axes[1, 0].set_title('5-Year Cost-Benefit Analysis')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Energy Reduction Roadmap
        phases = ['Current', 'Phase 1\n(Quick Wins)', 'Phase 2\n(Automation)', 
                 'Phase 3\n(Solar)']
        energy_usage = [100, 85, 60, 15]  # Percentage of current usage
        colors_phases = ['#95a5a6', '#3498db', '#f39c12', '#2ecc71']
        
        bars = axes[1, 1].bar(phases, energy_usage, color=colors_phases, alpha=0.7, edgecolor='black')
        axes[1, 1].set_ylabel('Energy Usage (% of Current)')
        axes[1, 1].set_title('Energy Reduction Roadmap')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        for bar, usage in zip(bars, energy_usage):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{usage}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('07_architectural_recommendations.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved: 07_architectural_recommendations.png")
        plt.show()
        
        # Calculate total potential savings
        total_savings_pct = 35  # Conservative estimate
        monthly_savings = (avg_power * 720 / 1000) * 8 * (total_savings_pct / 100)
        
        print(f"\n📊 PROJECTED IMPACT:")
        print(f"  Total Potential Energy Reduction: {total_savings_pct}%")
        print(f"  Monthly Cost Savings: ₹{monthly_savings:.2f}")
        print(f"  Annual Savings: ₹{monthly_savings * 12:.2f}")
        print(f"  3-Year Cumulative Savings: ₹{monthly_savings * 36:.2f}")
        
        # Save personalized report
        self.save_personalized_report(user_profile, recommendations, monthly_savings)
    
    def generate_personalized_recommendations(self, profile, avg_power, peak_power, current_pf):
        """Generate personalized recommendations based on user profile"""
        recommendations = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }
        
        # Analyze building age and insulation
        if profile['building_age'] in ['c', 'd'] and profile['insulation'] in ['b', 'c']:
            recommendations['high_priority'].append({
                'title': 'Thermal Insulation Upgrade',
                'reason': f"Your building is {'>15 years old' if profile['building_age'] == 'c' else '>30 years old'} with inadequate insulation",
                'action': 'Install roof insulation (R-30 rating) and wall insulation. Seal air leaks around doors and windows.',
                'cost': '₹80,000 - ₹1,50,000',
                'savings': '25-30% reduction in cooling costs',
                'payback': '2-3 years'
            })
        
        # Analyze lighting
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
        
        # Analyze cooling system
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
        
        # Analyze windows
        if profile['windows'] in ['b', 'c']:
            recommendations['medium_priority'].append({
                'title': 'Window Upgrade or Treatment',
                'reason': 'Single-pane or old windows cause significant heat gain',
                'action': 'Option 1: Install reflective window films (cheaper). Option 2: Upgrade to double-glazed windows (better long-term).',
                'cost': 'Films: ₹100-200/sq.ft, New windows: ₹600-1000/sq.ft',
                'savings': 'Films: 10-15%, New windows: 20-25% cooling savings',
                'payback': 'Films: 2-3 years, Windows: 5-7 years'
            })
        
        # Analyze power factor
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
        
        # Solar recommendations based on interest and budget
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
        
        # Building type specific recommendations
        if profile['building_type'] == 'b':  # Commercial
            recommendations['medium_priority'].append({
                'title': 'Occupancy-Based Automation',
                'reason': 'Commercial buildings benefit greatly from occupancy sensors',
                'action': 'Install PIR sensors for lighting and HVAC control in conference rooms, restrooms, and corridors.',
                'cost': '₹50,000 - ₹1,20,000',
                'savings': '25-35% on lighting and HVAC in low-occupancy areas',
                'payback': '2-3 years'
            })
        elif profile['building_type'] == 'a':  # Residential
            recommendations['low_priority'].append({
                'title': 'Smart Home Energy Management',
                'reason': 'Residential buildings benefit from smart controls and scheduling',
                'action': 'Install smart plugs, Wi-Fi switches, and energy monitoring system for major appliances.',
                'cost': '₹15,000 - ₹40,000',
                'savings': '10-15% through better control and phantom load elimination',
                'payback': '2-3 years'
            })
        
        # Budget-based quick wins
        if profile['budget'] == 'a':  # Under ₹50,000
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
        if consumption_per_sqft > 15:  # Wh per sq.ft per day
            recommendations['high_priority'].append({
                'title': 'Energy Audit - High Consumption Detected',
                'reason': f'Your consumption of {consumption_per_sqft:.1f} Wh/sq.ft/day is above normal ({">20%" if consumption_per_sqft > 18 else "10-20%"})',
                'action': 'Schedule professional energy audit to identify specific inefficiencies and ghost loads.',
                'cost': '₹5,000 - ₹15,000',
                'savings': 'Audit will reveal 20-30% saving opportunities',
                'payback': 'Immediate through identified savings'
            })
        
        # Voltage stabilization
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
        
        # Ensure we have at least one recommendation in each category
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
    
    def save_personalized_report(self, profile, recommendations, monthly_savings):
        """Save personalized recommendations to a text file"""
        try:
            with open('personalized_energy_report.txt', 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("PERSONALIZED ENERGY OPTIMIZATION REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("YOUR PROFILE:\n")
                f.write("-" * 80 + "\n")
                building_types = {'a': 'Residential', 'b': 'Commercial', 'c': 'Industrial', 'd': 'Mixed-use'}
                f.write(f"Building Type: {building_types.get(profile.get('building_type', 'a'), 'Residential')}\n")
                f.write(f"Floor Area: {profile.get('floor_area', 'N/A')} sq.ft\n")
                f.write(f"Daily Occupancy: {profile.get('occupancy_hours', 'N/A')} hours\n\n")
                
                f.write("HIGH PRIORITY RECOMMENDATIONS:\n")
                f.write("=" * 80 + "\n")
                for i, rec in enumerate(recommendations['high_priority'], 1):
                    f.write(f"\n{i}. {rec['title']}\n")
                    f.write(f"   Reason: {rec['reason']}\n")
                    f.write(f"   Action: {rec['action']}\n")
                    f.write(f"   Cost: {rec['cost']}\n")
                    f.write(f"   Savings: {rec['savings']}\n")
                    f.write(f"   Payback: {rec['payback']}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("MEDIUM PRIORITY RECOMMENDATIONS:\n")
                f.write("=" * 80 + "\n")
                for i, rec in enumerate(recommendations['medium_priority'], 1):
                    f.write(f"\n{i}. {rec['title']}\n")
                    f.write(f"   Reason: {rec['reason']}\n")
                    f.write(f"   Action: {rec['action']}\n")
                    f.write(f"   Cost: {rec['cost']}\n")
                    f.write(f"   Savings: {rec['savings']}\n")
                    f.write(f"   Payback: {rec['payback']}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"ESTIMATED IMPACT:\n")
                f.write(f"Monthly Savings Potential: ₹{monthly_savings:.2f}\n")
                f.write(f"Annual Savings Potential: ₹{monthly_savings * 12:.2f}\n")
                f.write(f"3-Year Cumulative Savings: ₹{monthly_savings * 36:.2f}\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("NEXT STEPS:\n")
                f.write("=" * 80 + "\n")
                f.write("1. Review and prioritize recommendations based on your budget\n")
                f.write("2. Get quotes from 3-4 vendors for major upgrades\n")
                f.write("3. Apply for government subsidies where applicable\n")
                f.write("4. Implement high-priority items within 3 months\n")
                f.write("5. Monitor energy consumption after each upgrade\n\n")
                
                f.write("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
                
            print("\n✓ Personalized report saved: personalized_energy_report.txt")
        except Exception as e:
            print(f"\n⚠️ Could not save personalized report: {e}")
    
    def generate_personalized_recommendations(self, profile, avg_power, peak_power, current_pf):
        """Generate personalized recommendations based on user profile"""
        recommendations = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }
        
        # Analyze building age and insulation
        if profile['building_age'] in ['c', 'd'] and profile['insulation'] in ['b', 'c']:
            recommendations['high_priority'].append({
                'title': 'Thermal Insulation Upgrade',
                'reason': f"Your building is {'>15 years old' if profile['building_age'] == 'c' else '>30 years old'} with inadequate insulation",
                'action': 'Install roof insulation (R-30 rating) and wall insulation. Seal air leaks around doors and windows.',
                'cost': '₹80,000 - ₹1,50,000',
                'savings': '25-30% reduction in cooling costs',
                'payback': '2-3 years'
            })
        
        # Analyze lighting
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
        
        # Analyze cooling system
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
        
        # Analyze windows
        if profile['windows'] in ['b', 'c']:
            recommendations['medium_priority'].append({
                'title': 'Window Upgrade or Treatment',
                'reason': 'Single-pane or old windows cause significant heat gain',
                'action': 'Option 1: Install reflective window films (cheaper). Option 2: Upgrade to double-glazed windows (better long-term).',
                'cost': 'Films: ₹100-200/sq.ft, New windows: ₹600-1000/sq.ft',
                'savings': 'Films: 10-15%, New windows: 20-25% cooling savings',
                'payback': 'Films: 2-3 years, Windows: 5-7 years'
            })
        
        # Analyze power factor
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
        
        # Solar recommendations based on interest and budget
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
        
        # Building type specific recommendations
        if profile['building_type'] == 'b':  # Commercial
            recommendations['medium_priority'].append({
                'title': 'Occupancy-Based Automation',
                'reason': 'Commercial buildings benefit greatly from occupancy sensors',
                'action': 'Install PIR sensors for lighting and HVAC control in conference rooms, restrooms, and corridors.',
                'cost': '₹50,000 - ₹1,20,000',
                'savings': '25-35% on lighting and HVAC in low-occupancy areas',
                'payback': '2-3 years'
            })
        elif profile['building_type'] == 'a':  # Residential
            recommendations['low_priority'].append({
                'title': 'Smart Home Energy Management',
                'reason': 'Residential buildings benefit from smart controls and scheduling',
                'action': 'Install smart plugs, Wi-Fi switches, and energy monitoring system for major appliances.',
                'cost': '₹15,000 - ₹40,000',
                'savings': '10-15% through better control and phantom load elimination',
                'payback': '2-3 years'
            })
        
        # Budget-based quick wins
        if profile['budget'] == 'a':  # Under ₹50,000
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
        if consumption_per_sqft > 15:  # Wh per sq.ft per day
            recommendations['high_priority'].append({
                'title': 'Energy Audit - High Consumption Detected',
                'reason': f'Your consumption of {consumption_per_sqft:.1f} Wh/sq.ft/day is above normal ({">20%" if consumption_per_sqft > 18 else "10-20%"})',
                'action': 'Schedule professional energy audit to identify specific inefficiencies and ghost loads.',
                'cost': '₹5,000 - ₹15,000',
                'savings': 'Audit will reveal 20-30% saving opportunities',
                'payback': 'Immediate through identified savings'
            })
        
        # Voltage stabilization
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
        
        # Ensure we have at least one recommendation in each category
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
    
    def save_personalized_report(self, profile, recommendations, monthly_savings):
        """Save personalized recommendations to a text file"""
        try:
            with open('personalized_energy_report.txt', 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("PERSONALIZED ENERGY OPTIMIZATION REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("YOUR PROFILE:\n")
                f.write("-" * 80 + "\n")
                building_types = {'a': 'Residential', 'b': 'Commercial', 'c': 'Industrial', 'd': 'Mixed-use'}
                f.write(f"Building Type: {building_types.get(profile.get('building_type', 'a'), 'Residential')}\n")
                f.write(f"Floor Area: {profile.get('floor_area', 'N/A')} sq.ft\n")
                f.write(f"Daily Occupancy: {profile.get('occupancy_hours', 'N/A')} hours\n\n")
                
                f.write("HIGH PRIORITY RECOMMENDATIONS:\n")
                f.write("=" * 80 + "\n")
                for i, rec in enumerate(recommendations['high_priority'], 1):
                    f.write(f"\n{i}. {rec['title']}\n")
                    f.write(f"   Reason: {rec['reason']}\n")
                    f.write(f"   Action: {rec['action']}\n")
                    f.write(f"   Cost: {rec['cost']}\n")
                    f.write(f"   Savings: {rec['savings']}\n")
                    f.write(f"   Payback: {rec['payback']}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("MEDIUM PRIORITY RECOMMENDATIONS:\n")
                f.write("=" * 80 + "\n")
                for i, rec in enumerate(recommendations['medium_priority'], 1):
                    f.write(f"\n{i}. {rec['title']}\n")
                    f.write(f"   Reason: {rec['reason']}\n")
                    f.write(f"   Action: {rec['action']}\n")
                    f.write(f"   Cost: {rec['cost']}\n")
                    f.write(f"   Savings: {rec['savings']}\n")
                    f.write(f"   Payback: {rec['payback']}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"ESTIMATED IMPACT:\n")
                f.write(f"Monthly Savings Potential: ₹{monthly_savings:.2f}\n")
                f.write(f"Annual Savings Potential: ₹{monthly_savings * 12:.2f}\n")
                f.write(f"3-Year Cumulative Savings: ₹{monthly_savings * 36:.2f}\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("NEXT STEPS:\n")
                f.write("=" * 80 + "\n")
                f.write("1. Review and prioritize recommendations based on your budget\n")
                f.write("2. Get quotes from 3-4 vendors for major upgrades\n")
                f.write("3. Apply for government subsidies where applicable\n")
                f.write("4. Implement high-priority items within 3 months\n")
                f.write("5. Monitor energy consumption after each upgrade\n\n")
                
                f.write("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
                
            print("\n✓ Personalized report saved: personalized_energy_report.txt")
        except Exception as e:
            print(f"\n⚠️ Could not save personalized report: {e}")
        
def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("AI-POWERED ENERGY ANALYSIS & OPTIMIZATION SYSTEM")
    print("=" * 70)
    
    # Initialize system
    try:
        analyzer = EnergyAnalysisSystem('energy_data.csv')
    except FileNotFoundError:
        print("\n❌ Error: 'energy_data.csv' not found!")
        print("Please ensure your data file is in the same directory as this script.")
        print("\nExpected columns: Voltage, Current, Power")
        return
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        return
    
    # Run all analyses
    print("\n" + "=" * 70)
    print("Running comprehensive energy analysis...")
    print("=" * 70)
    
    analyzer.ai_energy_prediction(hours_ahead=24)
    analyzer.batch_efficiency_analysis()
    analyzer.monthly_energy_report()
    analyzer.detect_peak_hours()
    analyzer.appliance_breakdown()
    analyzer.carbon_credit_tracking()
    analyzer.architectural_recommendations()
    
    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 70)
    print("\n📊 Generated Visualizations:")
    print("  1. 01_ai_prediction.png - AI prediction models and forecasts")
    print("  2. 02_efficiency_analysis.png - Efficiency and power factor analysis")
    print("  3. 03_monthly_report.png - Monthly consumption patterns")
    print("  4. 04_peak_hours.png - Peak hour detection and load analysis")
    print("  5. 05_appliance_breakdown.png - Appliance-level consumption")
    print("  6. 06_carbon_tracking.png - Carbon emissions and credits")
    print("  7. 07_architectural_recommendations.png - Upgrade recommendations")
    print("  8. personalized_energy_report.txt - Your customized report")
    print("\n📧 For detailed implementation plans, contact: energy@optimizer.com")
    print("🌐 Dashboard available at: https://energy-dashboard.example.com")
    
    analyzer.ai_energy_prediction(hours_ahead=24)
    analyzer.batch_efficiency_analysis()
    analyzer.monthly_energy_report()
    analyzer.detect_peak_hours()
    analyzer.appliance_breakdown()
    analyzer.carbon_credit_tracking()
    analyzer.architectural_recommendations()
    
    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 70)
    print("\n📊 Generated Visualizations:")
    print("  1. 01_ai_prediction.png - AI prediction models and forecasts")
    print("  2. 02_efficiency_analysis.png - Efficiency and power factor analysis")
    print("  3. 03_monthly_report.png - Monthly consumption patterns")
    print("  4. 04_peak_hours.png - Peak hour detection and load analysis")
    print("  5. 05_appliance_breakdown.png - Appliance-level consumption")
    print("  6. 06_carbon_tracking.png - Carbon emissions and credits")
    print("  7. 07_architectural_recommendations.png - Upgrade recommendations")
    print("  8. personalized_energy_report.txt - Your customized report")
    print("\n📧 For detailed implementation plans, contact: energy@optimizer.com")
    print("🌐 Dashboard available at: https://energy-dashboard.example.com")

if __name__ == "__main__":
    main()