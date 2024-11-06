// pages/settings.js
// frontend/pages/settings.js

import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';

export default function Settings() {
  return (
    <div className="dashboard">
      <Navbar />
      <div className="dashboard-content">
        <Sidebar />
        <div className="main-view">
          <h1>Settings</h1>
          <p>Configure your bot settings here.</p>
          {/* Add forms for settings */}
        </div>
      </div>
    </div>
  );
}
