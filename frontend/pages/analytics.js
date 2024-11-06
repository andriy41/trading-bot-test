// pages/analytics.js
// frontend/pages/analytics.js

import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';

export default function Analytics() {
  return (
    <div className="dashboard">
      <Navbar />
      <div className="dashboard-content">
        <Sidebar />
        <div className="main-view">
          <h1>Analytics</h1>
          <p>View bot performance analytics here.</p>
          {/* Add charts for metrics and performance analysis */}
        </div>
      </div>
    </div>
  );
}
