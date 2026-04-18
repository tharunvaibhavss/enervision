import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { Activity, Zap, TrendingUp, DollarSign, Leaf, Settings, AlertTriangle, CheckCircle, Download } from 'lucide-react';

const COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#95a5a6'];


const EnergyDashboard = () => {
  const [energyData, setEnergyData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [showQuestionnaire, setShowQuestionnaire] = useState(false);
  const [questionIndex, setQuestionIndex] = useState(0);
  const [userProfile, setUserProfile] = useState({});
  const [recommendations, setRecommendations] = useState(null);
  const [showReport, setShowReport] = useState(false);
  const intervalRef = useRef(null);

  // Questionnaire data
  const questions = [
    {
      id: 'building_type',
      question: 'What type of building is this?',
      options: [
        { value: 'a', label: 'Residential home' },
        { value: 'b', label: 'Commercial office' },
        { value: 'c', label: 'Industrial facility' },
        { value: 'd', label: 'Mixed-use building' }
      ]
    },
    {
      id: 'building_age',
      question: 'How old is your building?',
      options: [
        { value: 'a', label: 'Less than 5 years' },
        { value: 'b', label: '5-15 years' },
        { value: 'c', label: '15-30 years' },
        { value: 'd', label: 'More than 30 years' }
      ]
    },
    {
      id: 'floor_area',
      question: 'What is the approximate floor area? (in sq.ft)',
      type: 'number',
      placeholder: 'e.g., 1500'
    },
    {
      id: 'cooling_system',
      question: 'What cooling system do you currently use?',
      options: [
        { value: 'a', label: 'Central AC' },
        { value: 'b', label: 'Split AC units' },
        { value: 'c', label: 'Window AC units' },
        { value: 'd', label: 'Fans only' },
        { value: 'e', label: 'No cooling system' }
      ]
    },
    {
      id: 'insulation',
      question: 'Does your building have proper insulation?',
      options: [
        { value: 'a', label: 'Yes, well insulated' },
        { value: 'b', label: 'Partial insulation' },
        { value: 'c', label: 'No insulation' }
      ]
    },
    {
      id: 'windows',
      question: 'What type of windows do you have?',
      options: [
        { value: 'a', label: 'Double-glazed/energy-efficient' },
        { value: 'b', label: 'Single-pane glass' },
        { value: 'c', label: 'Old wooden/metal frame' }
      ]
    },
    {
      id: 'lighting',
      question: 'What percentage of your lighting is LED?',
      options: [
        { value: 'a', label: '80-100%' },
        { value: 'b', label: '40-80%' },
        { value: 'c', label: 'Less than 40%' }
      ]
    },
    {
      id: 'solar_interest',
      question: 'Are you interested in solar panel installation?',
      options: [
        { value: 'a', label: 'Yes, very interested' },
        { value: 'b', label: 'Maybe, want to know more' },
        { value: 'c', label: 'Not interested' }
      ]
    },
    {
      id: 'budget',
      question: 'What is your budget range for energy upgrades?',
      options: [
        { value: 'a', label: 'Under ₹50,000' },
        { value: 'b', label: '₹50,000 - ₹2,00,000' },
        { value: 'c', label: '₹2,00,000 - ₹5,00,000' },
        { value: 'd', label: 'Above ₹5,00,000' }
      ]
    },
    {
      id: 'occupancy_hours',
      question: 'How many hours per day is the building typically occupied?',
      type: 'number',
      placeholder: 'e.g., 12'
    },
    {
      id: 'current_bill',
      question: 'What is your approximate monthly electricity bill? (₹)',
      type: 'number',
      placeholder: 'e.g., 5000'
    },
    {
      id: 'priority',
      question: 'What is your main priority?',
      options: [
        { value: 'a', label: 'Reduce costs' },
        { value: 'b', label: 'Environmental impact' },
        { value: 'c', label: 'Energy independence' },
        { value: 'd', label: 'Equipment protection' }
      ]
    }
  ];

  // Fetch data from Google Sheets
  const fetchData = async () => {
    try {
      const sheetId = '1swNSIGLsCL-O_31tXhPXeyDoMntUin3hMyxTLVLvnr0';
      const url = `https://docs.google.com/spreadsheets/d/${sheetId}/gviz/tq?tqx=out:json`;
      
      const response = await fetch(url);
      const text = await response.text();
      const json = JSON.parse(text.substr(47).slice(0, -2));
      
      const data = json.table.rows.map((row, index) => ({
        index,
        timestamp: new Date(Date.now() - (json.table.rows.length - index) * 6000),
        voltage: row.c[0]?.v || 0,
        current: row.c[1]?.v || 0,
        power: row.c[2]?.v || 0,
        powerFactor: (row.c[2]?.v || 0) / ((row.c[0]?.v || 1) * (row.c[1]?.v || 1)),
        efficiency: ((row.c[2]?.v || 0) / ((row.c[0]?.v || 1) * (row.c[1]?.v || 1))) * 100
      }));
      
      setEnergyData(data);
      setLastUpdate(new Date());
      setIsLoading(false);
    } catch (error) {
      console.error('Error fetching data:', error);
      // Generate sample data if fetch fails
      generateSampleData();
    }
  };

  const generateSampleData = () => {
    const data = Array.from({ length: 180 }, (_, i) => ({
      index: i,
      timestamp: new Date(Date.now() - (180 - i) * 6000),
      voltage: 235 + Math.random() * 10,
      current: 3.5 + Math.random() * 0.5,
      power: 800 + Math.random() * 100,
      powerFactor: 0.998 + Math.random() * 0.004,
      efficiency: 99.8 + Math.random() * 0.4
    }));
    setEnergyData(data);
    setIsLoading(false);
  };

  useEffect(() => {
    fetchData();
    intervalRef.current = setInterval(fetchData, 10000); // Update every 10 seconds
    
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  // Calculate metrics
  const calculateMetrics = () => {
    if (energyData.length === 0) return null;
    
    const avgPower = energyData.reduce((sum, d) => sum + d.power, 0) / energyData.length;
    const avgVoltage = energyData.reduce((sum, d) => sum + d.voltage, 0) / energyData.length;
    const avgCurrent = energyData.reduce((sum, d) => sum + d.current, 0) / energyData.length;
    const avgPowerFactor = energyData.reduce((sum, d) => sum + d.powerFactor, 0) / energyData.length;
    const avgEfficiency = energyData.reduce((sum, d) => sum + d.efficiency, 0) / energyData.length;
    const totalEnergy = (avgPower * energyData.length * 6) / (1000 * 3600); // kWh
    const estimatedCost = totalEnergy * 8; // ₹8 per kWh
    const carbonEmissions = totalEnergy * 0.82; // kg CO2
    const peakPower = Math.max(...energyData.map(d => d.power));
    
    return {
      avgPower,
      avgVoltage,
      avgCurrent,
      avgPowerFactor,
      avgEfficiency,
      totalEnergy,
      estimatedCost,
      carbonEmissions,
      peakPower
    };
  };

  const metrics = calculateMetrics();

  // Generate recommendations
  const generateRecommendations = () => {
    if (!metrics) return;
    
    const recs = {
      high_priority: [],
      medium_priority: [],
      low_priority: []
    };

    // Power Factor Check
    if (metrics.avgPowerFactor < 0.85) {
      recs.high_priority.push({
        title: 'Power Factor Correction - URGENT',
        reason: `Your power factor is ${metrics.avgPowerFactor.toFixed(3)} - causing excess current draw`,
        action: 'Install automatic capacitor bank (APFC panel)',
        cost: '₹25,000 - ₹60,000',
        savings: `${((1 - metrics.avgPowerFactor/0.95) * 100).toFixed(1)}% current reduction`,
        payback: '12-18 months'
      });
    }

    // Building Age + Insulation
    if (userProfile.building_age === 'c' || userProfile.building_age === 'd') {
      if (userProfile.insulation === 'b' || userProfile.insulation === 'c') {
        recs.high_priority.push({
          title: 'Thermal Insulation Upgrade',
          reason: 'Old building with inadequate insulation',
          action: 'Install roof insulation (R-30) and seal air leaks',
          cost: '₹80,000 - ₹1,50,000',
          savings: '25-30% reduction in cooling costs',
          payback: '2-3 years'
        });
      }
    }

    // Lighting
    if (userProfile.lighting === 'c') {
      recs.high_priority.push({
        title: 'Complete LED Conversion',
        reason: 'Less than 40% LED - massive energy waste',
        action: 'Replace all bulbs with LED',
        cost: '₹15,000 - ₹40,000',
        savings: '75-80% lighting energy savings',
        payback: '8-12 months'
      });
    }

    // Cooling System
    if (userProfile.cooling_system === 'c') {
      recs.high_priority.push({
        title: 'Replace Window AC Units',
        reason: 'Window ACs are 40% less efficient',
        action: 'Upgrade to 5-star inverter split AC',
        cost: '₹35,000 - ₹50,000 per unit',
        savings: '40-45% cooling cost reduction',
        payback: '3-4 years'
      });
    }

    // Solar
    if (userProfile.solar_interest === 'a' && (userProfile.budget === 'c' || userProfile.budget === 'd')) {
      const solarKw = (metrics.avgPower * (userProfile.occupancy_hours || 12)) / 4000;
      recs.high_priority.push({
        title: 'Rooftop Solar Installation',
        reason: 'High interest + adequate budget',
        action: `Install ${solarKw.toFixed(1)} kW rooftop solar with net metering`,
        cost: `₹${(solarKw * 50000).toFixed(0)} - ₹${(solarKw * 60000).toFixed(0)}`,
        savings: `70-80% grid independence, ₹${(solarKw * 1500 * 8).toFixed(0)}/year`,
        payback: '4-5 years'
      });
    }

    // Windows
    if (userProfile.windows === 'b' || userProfile.windows === 'c') {
      recs.medium_priority.push({
        title: 'Window Upgrade or Treatment',
        reason: 'Single-pane windows cause heat gain',
        action: 'Install reflective films or upgrade to double-glazed',
        cost: 'Films: ₹100-200/sq.ft, Windows: ₹600-1000/sq.ft',
        savings: '10-25% cooling savings',
        payback: 'Films: 2-3 years, Windows: 5-7 years'
      });
    }

    // Smart Thermostat
    if (userProfile.cooling_system === 'a' || userProfile.cooling_system === 'b') {
      recs.medium_priority.push({
        title: 'Smart Thermostat Installation',
        reason: 'Optimize existing AC with intelligent control',
        action: 'Install Wi-Fi smart thermostats',
        cost: '₹5,000 - ₹15,000 per unit',
        savings: '15-20% on cooling costs',
        payback: '12-18 months'
      });
    }

    // Energy Monitoring
    recs.low_priority.push({
      title: 'Advanced Energy Monitoring',
      reason: 'Track and optimize usage patterns',
      action: 'Install real-time monitoring with mobile app',
      cost: '₹8,000 - ₹20,000',
      savings: '5-10% through behavioral changes',
      payback: '2-3 years'
    });

    setRecommendations(recs);
    setShowReport(true);
  };

  const handleQuestionAnswer = (answer) => {
    setUserProfile({ ...userProfile, [questions[questionIndex].id]: answer });
    
    if (questionIndex < questions.length - 1) {
      setQuestionIndex(questionIndex + 1);
    } else {
      setShowQuestionnaire(false);
      generateRecommendations();
    }
  };

  const downloadReport = () => {
    if (!recommendations || !metrics) return;
    
    let report = '='.repeat(80) + '\n';
    report += 'PERSONALIZED ENERGY OPTIMIZATION REPORT\n';
    report += '='.repeat(80) + '\n\n';
    
    report += 'CURRENT SYSTEM ANALYSIS:\n';
    report += '-'.repeat(80) + '\n';
    report += `Average Power: ${metrics.avgPower.toFixed(2)} W\n`;
    report += `Peak Power: ${metrics.peakPower.toFixed(2)} W\n`;
    report += `Power Factor: ${metrics.avgPowerFactor.toFixed(3)}\n`;
    report += `Efficiency: ${metrics.avgEfficiency.toFixed(2)}%\n`;
    report += `Total Energy: ${metrics.totalEnergy.toFixed(2)} kWh\n`;
    report += `Estimated Cost: ₹${metrics.estimatedCost.toFixed(2)}\n`;
    report += `Carbon Emissions: ${metrics.carbonEmissions.toFixed(2)} kg CO2\n\n`;
    
    report += 'HIGH PRIORITY RECOMMENDATIONS:\n';
    report += '='.repeat(80) + '\n';
    recommendations.high_priority.forEach((rec, i) => {
      report += `\n${i + 1}. ${rec.title}\n`;
      report += `   Reason: ${rec.reason}\n`;
      report += `   Action: ${rec.action}\n`;
      report += `   Cost: ${rec.cost}\n`;
      report += `   Savings: ${rec.savings}\n`;
      report += `   Payback: ${rec.payback}\n`;
    });
    
    report += '\n' + '='.repeat(80) + '\n';
    report += 'MEDIUM PRIORITY RECOMMENDATIONS:\n';
    report += '='.repeat(80) + '\n';
    recommendations.medium_priority.forEach((rec, i) => {
      report += `\n${i + 1}. ${rec.title}\n`;
      report += `   Reason: ${rec.reason}\n`;
      report += `   Action: ${rec.action}\n`;
      report += `   Cost: ${rec.cost}\n`;
      report += `   Savings: ${rec.savings}\n`;
      report += `   Payback: ${rec.payback}\n`;
    });
    
    report += '\nGenerated: ' + new Date().toLocaleString() + '\n';
    
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'energy_optimization_report.txt';
    a.click();
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center w-full h-screen bg-gray-900">
        <div className="text-center">
          <Activity className="text-blue-500 animate-spin w-full mb-6 p-0 m-0 mb-4" />
          <p className="text-white text-xl">Loading Energy Data...</p>
        </div>
      </div>
    );
  }

  if (showQuestionnaire) {
    const currentQ = questions[questionIndex];
    
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 p-0">
        <div className="w-full mb-6 p-0 m-0">
          <div className="bg-white rounded-lg shadow-2xl p-8">
            <div className="mb-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold text-gray-800">Energy Assessment</h2>
                <span className="text-sm text-gray-600">Question {questionIndex + 1} of {questions.length}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${((questionIndex + 1) / questions.length) * 100}%` }}
                />
              </div>
            </div>
            
            <h3 className="text-xl font-semibold text-gray-800 mb-6">{currentQ.question}</h3>
            
            {currentQ.type === 'number' ? (
              <div>
                <input
                  type="number"
                  placeholder={currentQ.placeholder}
                  className="w-full p-4 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none text-lg"
                  onKeyPress={(e) => {
                    if (e.key === 'Enter' && e.target.value) {
                      handleQuestionAnswer(parseFloat(e.target.value));
                    }
                  }}
                />
                <button
                  onClick={(e) => {
                    const input = e.target.previousSibling;
                    if (input.value) handleQuestionAnswer(parseFloat(input.value));
                  }}
                  className="mt-4 w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Continue
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                {currentQ.options.map((option) => (
                  <button
                    key={option.value}
                    onClick={() => handleQuestionAnswer(option.value)}
                    className="w-full p-4 text-left border-2 border-gray-300 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-all"
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            )}
            
            {questionIndex > 0 && (
              <button
                onClick={() => setQuestionIndex(questionIndex - 1)}
                className="mt-6 text-blue-600 hover:text-blue-800"
              >
                ← Previous Question
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (showReport && recommendations) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 p-8">
        <div className="max-w-6xl w-full mb-6 p-0 m-0">
          <div className="bg-white rounded-lg shadow-2xl p-8 mb-6">
            <div className="flex justify-between items-center mb-6">
              <h1 className="text-3xl font-bold text-gray-800">Your Personalized Energy Report</h1>
              <button
                onClick={downloadReport}
                className="flex items-center gap-2 bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700"
              >
                <Download className="w-5 h-5" />
                Download Report
              </button>
            </div>
            
            {/* System Analysis */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
              <div className="bg-blue-50 p-4 rounded-lg">
                <p className="text-sm text-gray-600">Avg Power</p>
                <p className="text-2xl font-bold text-blue-600">{metrics.avgPower.toFixed(0)}W</p>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <p className="text-sm text-gray-600">Power Factor</p>
                <p className="text-2xl font-bold text-green-600">{metrics.avgPowerFactor.toFixed(3)}</p>
              </div>
              <div className="bg-yellow-50 p-4 rounded-lg">
                <p className="text-sm text-gray-600">Total Energy</p>
                <p className="text-2xl font-bold text-yellow-600">{metrics.totalEnergy.toFixed(2)} kWh</p>
              </div>
              <div className="bg-red-50 p-4 rounded-lg">
                <p className="text-sm text-gray-600">CO2 Emissions</p>
                <p className="text-2xl font-bold text-red-600">{metrics.carbonEmissions.toFixed(2)} kg</p>
              </div>
            </div>

            {/* High Priority */}
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-red-600 mb-4 flex items-center gap-2">
                <AlertTriangle className="w-6 h-6" />
                HIGH PRIORITY (Immediate Action)
              </h2>
              <div className="space-y-4">
                {recommendations.high_priority.map((rec, i) => (
                  <div key={i} className="border-l-4 border-red-600 bg-red-50 p-4 rounded">
                    <h3 className="font-bold text-lg mb-2">{rec.title}</h3>
                    <p className="text-sm text-gray-700 mb-1"><strong>Why:</strong> {rec.reason}</p>
                    <p className="text-sm text-gray-700 mb-1"><strong>Action:</strong> {rec.action}</p>
                    <p className="text-sm text-gray-700 mb-1"><strong>Investment:</strong> {rec.cost}</p>
                    <p className="text-sm text-gray-700 mb-1"><strong>Savings:</strong> {rec.savings}</p>
                    <p className="text-sm text-gray-700"><strong>Payback:</strong> {rec.payback}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Medium Priority */}
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-yellow-600 mb-4 flex items-center gap-2">
                <Settings className="w-6 h-6" />
                MEDIUM PRIORITY (Plan within 6 months)
              </h2>
              <div className="space-y-4">
                {recommendations.medium_priority.map((rec, i) => (
                  <div key={i} className="border-l-4 border-yellow-600 bg-yellow-50 p-4 rounded">
                    <h3 className="font-bold text-lg mb-2">{rec.title}</h3>
                    <p className="text-sm text-gray-700 mb-1"><strong>Why:</strong> {rec.reason}</p>
                    <p className="text-sm text-gray-700 mb-1"><strong>Action:</strong> {rec.action}</p>
                    <p className="text-sm text-gray-700 mb-1"><strong>Investment:</strong> {rec.cost}</p>
                    <p className="text-sm text-gray-700 mb-1"><strong>Savings:</strong> {rec.savings}</p>
                    <p className="text-sm text-gray-700"><strong>Payback:</strong> {rec.payback}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Low Priority */}
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-green-600 mb-4 flex items-center gap-2">
                <CheckCircle className="w-6 h-6" />
                LOW PRIORITY (Future Consideration)
              </h2>
              <div className="space-y-4">
                {recommendations.low_priority.map((rec, i) => (
                  <div key={i} className="border-l-4 border-green-600 bg-green-50 p-4 rounded">
                    <h3 className="font-bold text-lg mb-2">{rec.title}</h3>
                    <p className="text-sm text-gray-700 mb-1"><strong>Why:</strong> {rec.reason}</p>
                    <p className="text-sm text-gray-700 mb-1"><strong>Action:</strong> {rec.action}</p>
                    <p className="text-sm text-gray-700 mb-1"><strong>Investment:</strong> {rec.cost}</p>
                    <p className="text-sm text-gray-700 mb-1"><strong>Savings:</strong> {rec.savings}</p>
                    <p className="text-sm text-gray-700"><strong>Payback:</strong> {rec.payback}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="flex gap-4">
              <button
                onClick={() => {
                  setShowReport(false);
                  setRecommendations(null);
                }}
                className="flex-1 bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700"
              >
                View Live Dashboard
              </button>
              <button
                onClick={() => {
                  setShowQuestionnaire(true);
                  setQuestionIndex(0);
                  setUserProfile({});
                  setShowReport(false);
                }}
                className="flex-1 bg-gray-600 text-white py-3 rounded-lg hover:bg-gray-700"
              >
                Retake Assessment
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Prepare chart data
  const recentData = energyData.slice(-30);
  const applianceData = [
    { name: 'Refrigerator/AC', value: 85, power: 905 },
    { name: 'Computer/TV', value: 8, power: 794 },
    { name: 'Base Load', value: 7, power: 4 }
  ];

  const hourlyData = energyData.reduce((acc, curr) => {
    const hour = curr.timestamp.getHours();
    if (!acc[hour]) acc[hour] = { hour, total: 0, count: 0 };
    acc[hour].total += curr.power;
    acc[hour].count++;
    return acc;
  }, {});

  const hourlyChart = Object.values(hourlyData).map(h => ({
    hour: h.hour,
    power: h.total / h.count
  })).sort((a, b) => a.hour - b.hour);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 p-4">
      {/* Header */}
      <div className="w-full mb-6 p-0 m-0 mb-6">
        <div className="bg-white rounded-lg shadow-xl p-6">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold text-gray-800 flex items-center gap-3">
                <Zap className="text-yellow-500" />
                Real-Time Energy Monitoring Dashboard
              </h1>
              <p className="text-gray-600 mt-1">
                Last updated: {lastUpdate.toLocaleTimeString()} | Auto-refresh: Every 10s
              </p>
            </div>
            <button
              onClick={() => setShowQuestionnaire(true)}
              className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-6 py-3 rounded-lg hover:from-blue-700 hover:to-purple-700 flex items-center gap-2 transition-all"
            >
              <Settings className="w-5 h-5" />
              Get Personalized Recommendations
            </button>
          </div>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="w-full mb-6 p-0 m-0 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg shadow-xl p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-blue-100 text-sm">Average Power</p>
                <p className="text-3xl font-bold">{metrics?.avgPower.toFixed(0)}W</p>
                <p className="text-blue-100 text-xs mt-1">Peak: {metrics?.peakPower.toFixed(0)}W</p>
              </div>
              <Activity className="w-12 h-12 text-blue-200" />
            </div>
          </div>

          <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-lg shadow-xl p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-green-100 text-sm">Efficiency</p>
                <p className="text-3xl font-bold">{metrics?.avgEfficiency.toFixed(1)}%</p>
                <p className="text-green-100 text-xs mt-1">PF: {metrics?.avgPowerFactor.toFixed(3)}</p>
              </div>
              <TrendingUp className="w-12 h-12 text-green-200" />
            </div>
          </div>

          <div className="bg-gradient-to-br from-yellow-500 to-orange-500 rounded-lg shadow-xl p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-yellow-100 text-sm">Estimated Cost</p>
                <p className="text-3xl font-bold">₹{metrics?.estimatedCost.toFixed(0)}</p>
                <p className="text-yellow-100 text-xs mt-1">{metrics?.totalEnergy.toFixed(2)} kWh</p>
              </div>
              <DollarSign className="w-12 h-12 text-yellow-200" />
            </div>
          </div>

          <div className="bg-gradient-to-br from-red-500 to-pink-500 rounded-lg shadow-xl p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-red-100 text-sm">Carbon Emissions</p>
                <p className="text-3xl font-bold">{metrics?.carbonEmissions.toFixed(1)}</p>
                <p className="text-red-100 text-xs mt-1">kg CO₂</p>
              </div>
              <Leaf className="w-12 h-12 text-red-200" />
            </div>
          </div>
        </div>
      </div>

      {/* Main Charts Grid */}
      <div className="w-full mb-6 p-0 m-0 space-y-6">
        
        {/* Row 1: Real-time Power & Voltage/Current */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-xl p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Real-Time Power Consumption</h2>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={recentData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={(time) => time.toLocaleTimeString().slice(0, 5)}
                  tick={{ fontSize: 12 }}
                />
                <YAxis />
                <Tooltip 
                  labelFormatter={(time) => time.toLocaleTimeString()}
                  formatter={(value) => [`${value.toFixed(2)} W`, 'Power']}
                />
                <Area type="monotone" dataKey="power" stroke="#3498db" fill="#3498db" fillOpacity={0.6} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-lg shadow-xl p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Voltage & Current Trends</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={recentData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={(time) => time.toLocaleTimeString().slice(0, 5)}
                  tick={{ fontSize: 12 }}
                />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip 
                  labelFormatter={(time) => time.toLocaleTimeString()}
                />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="voltage" stroke="#e74c3c" strokeWidth={2} name="Voltage (V)" />
                <Line yAxisId="right" type="monotone" dataKey="current" stroke="#2ecc71" strokeWidth={2} name="Current (A)" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Row 2: Efficiency & Power Factor */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-xl p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">System Efficiency Over Time</h2>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={recentData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={(time) => time.toLocaleTimeString().slice(0, 5)}
                  tick={{ fontSize: 12 }}
                />
                <YAxis domain={[95, 105]} />
                <Tooltip 
                  labelFormatter={(time) => time.toLocaleTimeString()}
                  formatter={(value) => [`${value.toFixed(2)}%`, 'Efficiency']}
                />
                <Area type="monotone" dataKey="efficiency" stroke="#2ecc71" fill="#2ecc71" fillOpacity={0.4} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-lg shadow-xl p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Power Factor Distribution</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={recentData.reduce((acc, curr) => {
                const pf = curr.powerFactor.toFixed(3);
                const existing = acc.find(a => a.pf === pf);
                if (existing) existing.count++;
                else acc.push({ pf, count: 1 });
                return acc;
              }, []).slice(0, 20)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="pf" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#3498db" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Row 3: Hourly Pattern & Load Duration */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-xl p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">24-Hour Load Profile</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={hourlyChart}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" label={{ value: 'Hour of Day', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Power (W)', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => [`${value.toFixed(2)} W`, 'Avg Power']} />
                <Bar dataKey="power" fill="#e74c3c" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-lg shadow-xl p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Load Duration Curve</h2>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={energyData.map(d => d.power).sort((a, b) => b - a).map((p, i) => ({ 
                percentile: (i / energyData.length) * 100, 
                power: p 
              }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="percentile" label={{ value: 'Percentage of Time (%)', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Power (W)', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => [`${value.toFixed(2)} W`, 'Power']} />
                <Area type="monotone" dataKey="power" stroke="#2ecc71" fill="#2ecc71" fillOpacity={0.6} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Row 4: Appliance Breakdown & Cost Analysis */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-xl p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Appliance Consumption Breakdown</h2>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={applianceData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => `${name}: ${value}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {applianceData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-lg shadow-xl p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Average Power by Category</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={applianceData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" label={{ value: 'Power (W)', position: 'insideBottom', offset: -5 }} />
                <YAxis dataKey="name" type="category" width={120} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="power" fill="#f39c12" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Row 5: Carbon Emissions & Cost Scenarios */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-xl p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Cumulative Carbon Emissions</h2>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={energyData.map((d, i) => ({
                index: i,
                emissions: energyData.slice(0, i + 1).reduce((sum, curr) => sum + (curr.power * 6 / (1000 * 3600)) * 0.82, 0)
              }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="index" label={{ value: 'Reading Number', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'CO₂ (kg)', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => [`${value.toFixed(3)} kg`, 'CO₂']} />
                <Area type="monotone" dataKey="emissions" stroke="#e74c3c" fill="#e74c3c" fillOpacity={0.6} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-lg shadow-xl p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Carbon Reduction Scenarios</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={[
                { scenario: 'Current', emissions: metrics?.carbonEmissions || 0, color: '#e74c3c' },
                { scenario: '15% Reduction', emissions: (metrics?.carbonEmissions || 0) * 0.85, color: '#f39c12' },
                { scenario: '30% Reduction', emissions: (metrics?.carbonEmissions || 0) * 0.70, color: '#f1c40f' },
                { scenario: 'With Solar', emissions: (metrics?.carbonEmissions || 0) * 0.05, color: '#2ecc71' }
              ]}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="scenario" tick={{ fontSize: 11 }} />
                <YAxis label={{ value: 'CO₂ (kg)', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => [`${value.toFixed(2)} kg`, 'Emissions']} />
                <Bar dataKey="emissions">
                  {[
                    { scenario: 'Current', emissions: metrics?.carbonEmissions || 0, color: '#e74c3c' },
                    { scenario: '15% Reduction', emissions: (metrics?.carbonEmissions || 0) * 0.85, color: '#f39c12' },
                    { scenario: '30% Reduction', emissions: (metrics?.carbonEmissions || 0) * 0.70, color: '#f1c40f' },
                    { scenario: 'With Solar', emissions: (metrics?.carbonEmissions || 0) * 0.05, color: '#2ecc71' }
                  ].map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Row 6: Energy Savings & ROI */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-xl p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Potential Savings by Upgrade</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={[
                { upgrade: 'Voltage Stabilizer', savings: 10 },
                { upgrade: 'Power Factor', savings: 8 },
                { upgrade: 'LED Conversion', savings: 15 },
                { upgrade: 'Smart Automation', savings: 25 },
                { upgrade: 'Solar PV', savings: 70 }
              ]} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" label={{ value: 'Energy Savings (%)', position: 'insideBottom', offset: -5 }} />
                <YAxis dataKey="upgrade" type="category" width={130} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="savings">
                  {[10, 8, 15, 25, 70].map((value, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-lg shadow-xl p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Return on Investment Timeline</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={[
                { upgrade: 'AVR', months: 21 },
                { upgrade: 'Capacitors', months: 18 },
                { upgrade: 'LED', months: 12 },
                { upgrade: 'IoT System', months: 36 },
                { upgrade: 'Solar PV', months: 54 }
              ]}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="upgrade" tick={{ fontSize: 11 }} />
                <YAxis label={{ value: 'Months', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => [`${value} months`, 'Payback']} />
                <Bar dataKey="months">
                  {[21, 18, 12, 36, 54].map((value, index) => (
                    <Cell key={`cell-${index}`} fill={value > 36 ? '#e74c3c' : value > 24 ? '#f39c12' : '#2ecc71'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Row 7: Energy Roadmap & Credit Revenue */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-xl p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Energy Reduction Roadmap</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={[
                { phase: 'Current', usage: 100 },
                { phase: 'Phase 1\n(Quick Wins)', usage: 85 },
                { phase: 'Phase 2\n(Automation)', usage: 60 },
                { phase: 'Phase 3\n(Solar)', usage: 15 }
              ]}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="phase" tick={{ fontSize: 11 }} />
                <YAxis label={{ value: 'Energy Usage (% of Current)', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => [`${value}%`, 'Usage']} />
                <Bar dataKey="usage">
                  {[100, 85, 60, 15].map((value, index) => (
                    <Cell key={`cell-${index}`} fill={['#95a5a6', '#3498db', '#f39c12', '#2ecc71'][index]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-lg shadow-xl p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Carbon Credit Revenue Projection</h2>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={[
                { period: 'Month 1', value: (metrics?.carbonEmissions || 0) * 0.15 / 1000 * 1200 },
                { period: 'Month 6', value: (metrics?.carbonEmissions || 0) * 0.15 / 1000 * 1200 * 6 },
                { period: 'Month 12', value: (metrics?.carbonEmissions || 0) * 0.15 / 1000 * 1200 * 12 },
                { period: 'Year 2', value: (metrics?.carbonEmissions || 0) * 0.15 / 1000 * 1200 * 24 },
                { period: 'Year 3', value: (metrics?.carbonEmissions || 0) * 0.15 / 1000 * 1200 * 36 }
              ]}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="period" tick={{ fontSize: 11 }} />
                <YAxis label={{ value: 'Cumulative Value (₹)', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => [`₹${value.toFixed(0)}`, 'Revenue']} />
                <Area type="monotone" dataKey="value" stroke="#2ecc71" fill="#2ecc71" fillOpacity={0.6} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

      {/* Footer */}
      <div className="w-full mb-6 p-0 m-0 mt-8 mb-4">
        <div className="bg-gray-800 rounded-lg shadow-xl p-6 text-center text-white">
          <p className="text-lg mb-2">
            🔋 Total Data Points: {energyData.length} | 📊 Update Frequency: Real-time (10s) | 🌍 Carbon Tracking: Active
          </p>
          <p className="text-sm text-gray-400">
            Dashboard updates automatically from your Google Sheets data source
          </p>
        </div>
      </div>
    </div>
  );
};



export default EnergyDashboard;