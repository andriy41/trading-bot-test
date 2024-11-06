// components/Sidebar.js
// frontend/components/Sidebar.js

import Link from 'next/link';

export default function Sidebar() {
  return (
    <div className="sidebar">
      <ul>
        <li><Link href="/">Dashboard</Link></li>
        <li><Link href="/settings">Settings</Link></li>
        <li><Link href="/analytics">Analytics</Link></li>
      </ul>
    </div>
  );
}
