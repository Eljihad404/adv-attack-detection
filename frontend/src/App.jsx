import { useState, useRef } from 'react'
import { analyzeImage } from './api'
import './index.css'

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [useDetector, setUseDetector] = useState(true);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) processFile(selectedFile);
  };

  const processFile = (selectedFile) => {
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError(null);
  };

  const handleDragOver = (e) => e.preventDefault();

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files[0]) processFile(e.dataTransfer.files[0]);
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeImage(file, useDetector);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      {/* Medical Header */}
      <header className="medical-header">
        <div style={{ fontSize: '2rem' }}>🏥</div>
        <div>
          <div className="brand-title">MediSecure Diagnostic</div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>AI-Powered Dept. of Radiology</div>
        </div>
      </header>

      <div className="main-container">

        {/* Left Column: Input */}
        <div className="clinical-card">
          <div className="card-title">PATIENT IMAGING INPUT (DICOM/JPEG)</div>

          <div
            className="upload-area"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
              style={{ display: 'none' }}
            />
            {preview ? (
              <img src={preview} alt="X-Ray Scan" style={{ maxWidth: '100%', maxHeight: '400px', objectFit: 'contain' }} />
            ) : (
              <>
                <div style={{ fontSize: '2.5rem', color: '#cbd5e1', marginBottom: '1rem' }}>📷</div>
                <div style={{ color: 'var(--text-muted)' }}>Click to Upload Chest X-Ray</div>
                <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.5rem' }}>Drag & Drop Supported</div>
              </>
            )}
          </div>

          <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem' }}>
            <input
              type="checkbox"
              id="useDetector"
              checked={useDetector}
              onChange={(e) => setUseDetector(e.target.checked)}
              style={{ width: '1.2rem', height: '1.2rem', cursor: 'pointer' }}
            />
            <label htmlFor="useDetector" style={{ cursor: 'pointer', userSelect: 'none' }}>
              Enable Adversarial Attack Detector
            </label>
          </div>

          <div style={{ borderTop: '1px solid var(--border)', marginTop: '1rem', paddingTop: '1rem' }}>
            <button
              className="btn-primary"
              disabled={!file || loading}
              onClick={handleSubmit}
            >
              {loading ? 'PROCESSING ANALYSIS...' : 'RUN DIAGNOSTIC ANALYSIS'}
            </button>
          </div>
        </div>

        {/* Right Column: Report */}
        <div className="clinical-card">
          <div className="card-title">DIAGNOSTIC REPORT</div>

          {error && (
            <div style={{ background: '#fee2e2', color: '#991b1b', padding: '1rem', borderRadius: '6px', fontSize: '0.9rem' }}>
              <strong>SYSTEM ERROR:</strong> {error}
            </div>
          )}

          {!result && !loading && !error && (
            <div style={{ textAlign: 'center', padding: '4rem 0', color: '#cbd5e1' }}>
              <div>📋</div>
              <p>No analysis data available.<br />Please execute scan.</p>
            </div>
          )}

          {loading && (
            <div style={{ textAlign: 'center', padding: '4rem 0' }}>
              <div style={{ marginBottom: '1rem', color: 'var(--primary)' }}>Running AI Inference Protocols...</div>
              <div style={{ width: '40px', height: '40px', border: '4px solid #e2e8f0', borderTopColor: 'var(--primary)', borderRadius: '50%', margin: '0 auto', animation: 'spin 1s linear infinite' }}></div>
              <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
            </div>
          )}

          {result && !loading && (
            <div className="fade-in">
              {/* Security Check Section */}
              {useDetector && (
                <div style={{ marginBottom: '2rem' }}>
                  <div className="metric-row" style={{ fontWeight: 'bold' }}>
                    <span>SECURITY INTEGRITY CHECK</span>
                    <span style={{ color: result.is_adversarial ? '#dc2626' : '#16a34a' }}>
                      {result.is_adversarial ? 'FAILED' : 'PASSED'}
                    </span>
                  </div>
                  {result.is_adversarial && (
                    <div style={{ background: '#fef2f2', border: '1px solid #fecaca', padding: '1rem', borderRadius: '6px', marginBottom: '1rem' }}>
                      <strong style={{ color: '#dc2626' }}>⚠️ ADVERSARIAL ATTACK DETECTED</strong>
                      <div style={{ fontSize: '0.875rem', marginTop: '0.5rem', color: '#7f1d1d' }}>
                        Image data integrity compromised. Perturbations detected.
                        Diagnostic halted for safety.
                      </div>
                    </div>
                  )}
                </div>
              )}


              {!result.is_adversarial && (
                <>
                  <div className="report-header">
                    <div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Classification</div>
                      <div style={{ fontSize: '1.5rem', fontWeight: '800', color: 'var(--text-main)' }}>
                        {result.prediction}
                      </div>
                    </div>
                    <div className={`status-badge ${result.prediction === 'PNEUMONIA' ? 'status-critical' : 'status-normal'}`}>
                      {result.prediction === 'NORMAL' ? 'NEGATIVE' : 'POSITIVE'}
                    </div>
                  </div>

                  <div style={{ marginTop: '1.5rem' }}>
                    <div style={{ fontSize: '0.85rem', fontWeight: '600', color: 'var(--text-muted)', marginBottom: '1rem' }}>CONFIDENCE METRICS</div>

                    {Object.entries(result.all_probabilities).map(([label, prob]) => (
                      <div key={label}>
                        <div className="metric-row">
                          <span>{label}</span>
                          <span>{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="progress-track">
                          <div
                            className="progress-fill"
                            style={{
                              width: `${prob * 100}%`,
                              backgroundColor: label === 'PNEUMONIA' ? '#ef4444' : '#22c55e',
                              opacity: result.prediction === label ? 1 : 0.3
                            }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}

              <div className="disclaimer">
                <strong>DISCLAIMER:</strong> This is an AI-assisted diagnostic tool. Results should be verified by a certified radiologist.
                <br />Device ID: MG-2026-X1 | Model: ResNet-Secure-v2
              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  )
}

export default App
