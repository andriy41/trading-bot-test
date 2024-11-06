// pages/index.js
// frontend/pages/index.js

import Navbar from '../components/Navbar';
import SignalChart from '../components/SignalChart';
import Sidebar from '../components/Sidebar';

export default function Dashboard() {
  return (
    <div className="dashboard">
      <Navbar />
      <div className="dashboard-content">
        <Sidebar />
        <div className="main-view">
          <h1>Trading Bot Dashboard</h1>
          <SignalChart />
        </div>
      </div>
    </div>
  );
}
