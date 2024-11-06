// components/SignalChart.js
// frontend/components/SignalChart.js

import dynamic from 'next/dynamic';
import { useEffect, useState } from 'react';
import axios from 'axios';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export default function SignalChart() {
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    // Fetch data from the backend
    async function fetchData() {
      const response = await axios.get('http://localhost:5000/api/test'); // Replace with actual endpoint
      setChartData(response.data);
    }
    fetchData();
  }, []);

  return (
    <div>
      <h3>Price and Signal Chart</h3>
      <Plot
        data={[
          {
            x: chartData.map(d => d.timestamp),
            y: chartData.map(d => d.price),
            type: 'scatter',
            mode: 'lines+markers',
            marker: { color: 'blue' },
          },
        ]}
        layout={{ title: 'Stock Price and Signals' }}
      />
    </div>
  );
}
