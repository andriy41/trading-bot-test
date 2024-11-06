// components/AlertPopup.js
// frontend/components/AlertPopup.js

import { useState, useEffect } from 'react';

export default function AlertPopup({ message }) {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setVisible(false), 3000);
    return () => clearTimeout(timer);
  }, []);

  if (!visible) return null;

  return (
    <div className="alert-popup">
      <p>{message}</p>
    </div>
  );
}
