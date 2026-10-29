import { useState, useCallback, useRef } from 'react'
import { MODALITY_MAP, API_BASE } from './config.js'
import ModalitySelector from './components/ModalitySelector.jsx'
import PipelineProgress from './components/PipelineProgress.jsx'
import ResultsPanel from './components/ResultsPanel.jsx'

/* ── helpers ──────────────────────────────────────────────── */
function nowStr() {
  const d = new Date()
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`
}

/* ═══════════════════════════════════════════════════════════
   App
════════════════════════════════════════════════════════════ */
export default function App() {
  const [selected, setSelected] = useState(null)      // 'text' | 'voice' | 'image'
  const [running, setRunning] = useState(false)
  const [logs, setLogs] = useState([])
  const [percent, setPercent] = useState(0)
  const [currentStage, setStage] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [backendOk, setBackendOk] = useState(null)      // null=unchecked, true, false
  const [view, setView] = useState('pipeline')         // 'pipeline' | 'detection'

  // Detection states
  const [detectText, setDetectText] = useState('')
  const [detectFile, setDetectFile] = useState(null)
  const [detectResult, setDetectResult] = useState(null)
  const [detecting, setDetecting] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [webcamActive, setWebcamActive] = useState(false)

  const esRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const videoRef = useRef(null)
  const canvasRef = useRef(null)

  /* ── Append to log feed ─────────────────────────────────── */
  const addLog = useCallback((msg) => {
    setLogs(prev => [...prev, { time: nowStr(), message: msg }])
  }, [])

  /* ── Health check ────────────────────────────────────────── */
  const checkBackend = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(4000) })
      setBackendOk(r.ok)
      return r.ok
    } catch {
      setBackendOk(false)
      return false
    }
  }, [])

  /* ── Modality select ─────────────────────────────────────── */
  const handleSelect = useCallback((id) => {
    if (running || detecting) return
    setSelected(id)
    setResult(null)
    setError(null)
    setLogs([])
    setPercent(0)
    setStage('')
    setDetectResult(null)
    setDetectFile(null)
    setDetectText('')
    checkBackend()
  }, [running, detecting, checkBackend])

  /* ── Run pipeline ────────────────────────────────────────── */
  const handleRun = useCallback(async () => {
    if (!selected || running) return

    // Close existing SSE
    if (esRef.current) { esRef.current.close(); esRef.current = null }

    setRunning(true)
    setResult(null)
    setError(null)
    setLogs([])
    setPercent(0)
    setStage('init')

    const mod = MODALITY_MAP[selected]
    addLog(`🚀 Starting ${mod.label} pipeline…`)

    try {
      // 1. POST to start job
      const postRes = await fetch(`${API_BASE}/api/run/${selected}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })

      if (!postRes.ok) {
        const txt = await postRes.text()
        throw new Error(`Backend returned ${postRes.status}: ${txt}`)
      }

      const { job_id } = await postRes.json()
      addLog(`📋 Job ID: ${job_id}`)

      // 2. Subscribe to SSE stream
      const es = new EventSource(`${API_BASE}/api/stream/${job_id}`)
      esRef.current = es

      es.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data)

          if (data.type === 'heartbeat') return

          if (data.type === 'progress') {
            addLog(data.message)
            if (data.percent !== undefined) setPercent(data.percent)
            if (data.stage) setStage(data.stage)
          }

          if (data.type === 'result') {
            addLog(`🎉 Pipeline complete! Accuracy: ${data.accuracy}%`)
            setPercent(100)
            setResult(data)
            setRunning(false)
            es.close()
          }

          if (data.type === 'error') {
            addLog(`❌ Error: ${data.message}`)
            setError({ message: data.message, traceback: data.traceback })
            setRunning(false)
            es.close()
          }
        } catch (parseErr) {
          console.error('SSE parse error', parseErr)
        }
      }

      es.onerror = () => {
        if (running) {
          addLog('⚠️ Connection lost — check backend is running.')
          setError({ message: 'SSE connection dropped. Is the Flask backend running on port 5050?' })
          setRunning(false)
        }
        es.close()
      }

    } catch (err) {
      addLog(`❌ ${err.message}`)
      setError({ message: err.message })
      setRunning(false)
    }
  }, [selected, running, addLog])

  /* ── Detection logic ─────────────────────────────────────── */
  const handleDetect = useCallback(async () => {
    if (!selected || detecting) return
    setDetecting(true)
    setDetectResult(null)
    setError(null)

    try {
      let res;
      if (selected === 'text') {
        res = await fetch(`${API_BASE}/api/detect/text`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: detectText })
        })
      } else if (selected === 'voice') {
        const formData = new FormData()
        if (detectFile) {
          formData.append('file', detectFile)
        } else {
          throw new Error('Please record or upload an audio file first.')
        }
        res = await fetch(`${API_BASE}/api/detect/voice`, {
          method: 'POST',
          body: formData
        })
      } else if (selected === 'image') {
        let body;
        let headers = {};
        if (webcamActive) {
          const canvas = canvasRef.current
          const video = videoRef.current
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
          canvas.getContext('2d').drawImage(video, 0, 0)
          const base64 = canvas.toDataURL('image/jpeg')
          body = JSON.stringify({ image: base64 })
          headers = { 'Content-Type': 'application/json' }
        } else if (detectFile) {
          const formData = new FormData()
          formData.append('file', detectFile)
          body = formData
        } else {
          throw new Error('Please capture or upload an image first.')
        }

        res = await fetch(`${API_BASE}/api/detect/image`, {
          method: 'POST',
          headers: headers,
          body: body
        })
      }

      if (!res.ok) {
        const errData = await res.json()
        throw new Error(errData.error || 'Detection failed')
      }

      const data = await res.json()
      setDetectResult(data)  // store full object: emotion, confidence, all_scores
    } catch (err) {
      setError({ message: err.message })
    } finally {
      setDetecting(false)
    }
  }, [selected, detecting, detectText, detectFile, webcamActive])

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      mediaRecorderRef.current = recorder
      audioChunksRef.current = []

      recorder.ondataavailable = (e) => audioChunksRef.current.push(e.data)
      recorder.onstop = () => {
        // Browser MediaRecorder always records webm/opus — do NOT label it as wav
        const mimeType = recorder.mimeType || 'audio/webm'
        const blob = new Blob(audioChunksRef.current, { type: mimeType })
        const ext = mimeType.includes('ogg') ? 'ogg' : mimeType.includes('mp4') ? 'mp4' : 'webm'
        setDetectFile(new File([blob], `recording.${ext}`, { type: mimeType }))
      }

      recorder.start()
      setIsRecording(true)
    } catch (err) {
      setError({ message: 'Could not access microphone: ' + err.message })
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  const toggleWebcam = async () => {
    if (webcamActive) {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(t => t.stop())
      }
      setWebcamActive(false)
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true })
        if (videoRef.current) videoRef.current.srcObject = stream
        setWebcamActive(true)
      } catch (err) {
        setError({ message: 'Could not access camera: ' + err.message })
      }
    }
  }

  /* ── Derived ─────────────────────────────────────────────── */
  const mod = selected ? MODALITY_MAP[selected] : null

  const btnLabel = running
    ? `Running ${mod?.shortLabel} Pipeline…`
    : selected
      ? `▶  Run ${mod?.label}`
      : 'Select a modality above'

  const btnClass = running || !selected
    ? 'run-btn run-btn-default'
    : `run-btn run-btn-${selected}`

  const showProgress = selected && (running || logs.length > 0)
  const showResults = result && !running
  const showIdle = !selected || (!running && logs.length === 0 && !result && !error)
  const showError = !!error

  /* ── Render ─────────────────────────────────────────────── */
  return (
    <div className="app-shell">

      {/* ── Header ───────────────────────────────────────── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <div className="logo-icon">🧠</div>
            <span className="logo-text">Emoti<span>Sense</span></span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            {backendOk !== null && (
              <span style={{
                display: 'flex', alignItems: 'center', gap: 6,
                fontSize: '0.72rem', color: backendOk ? 'var(--green)' : 'var(--red)'
              }}>
                <span style={{
                  width: 7, height: 7, borderRadius: '50%',
                  background: backendOk ? 'var(--green)' : 'var(--red)',
                  display: 'inline-block',
                  animation: backendOk ? 'pulse-dot 2s infinite' : 'none'
                }} />
                {backendOk ? 'Backend Connected' : 'Backend Offline'}
              </span>
            )}
            <span className="header-badge">ML Dashboard</span>
          </div>
        </div>
      </header>

      {/* ── Main grid ────────────────────────────────────────── */}
      <main className="main-content">

        {/* LEFT — selector + run button */}
        <aside>
          <div className="panel" style={{ marginBottom: 16 }}>
            <p className="panel-title">Select Modality</p>
            <ModalitySelector selected={selected} onSelect={handleSelect} />

            <button
              id="run-pipeline-btn"
              className={btnClass}
              onClick={handleRun}
              disabled={!selected || running}
              style={{ marginTop: 20 }}
              aria-label={btnLabel}
            >
              <span className="run-btn-inner">
                {running && <span className="spinner" />}
                {btnLabel}
              </span>
            </button>
          </div>

          {/* Quick info card */}
          {mod && !running && (
            <div className="panel" style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.7 }}>
              <p className="panel-title">Pipeline Info</p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <Row label="Model" value={
                  selected === 'text' ? 'Logistic Regression' :
                    selected === 'voice' ? 'Random Forest (200 trees)' :
                      'CNN (3 conv blocks)'
                } />
                <Row label="Features" value={
                  selected === 'text' ? 'TF-IDF (5000 features)' :
                    selected === 'voice' ? 'MFCC (40 coefficients)' :
                      'Raw pixels 48×48 grayscale'
                } />
                <Row label="Library" value={
                  selected === 'text' ? 'scikit-learn + NLTK' :
                    selected === 'voice' ? 'librosa + scikit-learn' :
                      'TensorFlow / Keras'
                } />
                <Row label="Dataset" value={
                  selected === 'text' ? 'Mental Health Dataset.csv' :
                    selected === 'voice' ? 'TESS / OAF Voice Corpus' :
                      'FER-style (train/test folders)'
                } />
              </div>
            </div>
          )}
        </aside>

        {/* RIGHT — progress + results */}
        <section className="right-panel">

          <div className="view-tabs">
            <button className={`view-tab ${view === 'pipeline' ? 'active' : ''}`} onClick={() => setView('pipeline')}>Pipeline</button>
            <button className={`view-tab ${view === 'detection' ? 'active' : ''}`} onClick={() => setView('detection')}>Real-time Detection</button>
          </div>

          {view === 'pipeline' ? (
            <>
              {/* Idle placeholder */}
              {showIdle && (
                <div className="panel" style={{ flex: 1 }}>
                  <div className="idle-state">
                    <div className="idle-icon">🧠</div>
                    <p className="idle-title">Ready to Train Models</p>
                    <p className="idle-sub">
                      Select a modality on the left — Text, Voice, or Facial Expression — then
                      click <strong>Run</strong> to execute the full training &amp; evaluation pipeline.
                    </p>
                    <div style={{ display: 'flex', gap: 12, marginTop: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
                      {['💬 Text NLP', '🎙️ Voice Audio', '😊 Facial CNN'].map(t => (
                        <span key={t} className="tag tag-neutral" style={{ fontSize: '0.75rem', padding: '6px 14px' }}>{t}</span>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Progress feed */}
              {showProgress && selected && (
                <PipelineProgress
                  modality={selected}
                  running={running}
                  logs={logs}
                  percent={percent}
                  currentStage={currentStage}
                />
              )}

              {/* Results */}
              {showResults && (
                <ResultsPanel result={result} modality={selected} />
              )}
            </>
          ) : (
            <div className="panel" style={{ flex: 1 }}>
              {!selected ? (
                <div className="idle-state">
                  <div className="idle-icon">✨</div>
                  <p className="idle-title">Select a modality to start detecting</p>
                  <p className="idle-sub">Choose between Text, Voice, or Image on the left panel to begin.</p>
                </div>
              ) : (
                <div className="detect-container">
                  <header style={{ marginBottom: 10 }}>
                    <h2 style={{ fontSize: '1.2rem', fontWeight: 700 }}>{MODALITY_MAP[selected].label} Detection</h2>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Enter input below to predict emotion in real-time.</p>
                  </header>

                  <div className="detect-input-area">
                    {selected === 'text' && (
                      <textarea
                        className="detect-textarea"
                        placeholder="Type something here (e.g., 'I am feeling very happy today!')..."
                        value={detectText}
                        onChange={(e) => setDetectText(e.target.value)}
                      />
                    )}

                    {selected === 'voice' && (
                      <div className="voice-detect-area">
                        <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
                          {!isRecording ? (
                            <button className="detect-btn" onClick={startRecording} style={{ background: 'var(--red)' }}>
                              🎙️ Start Recording
                            </button>
                          ) : (
                            <button className="detect-btn" onClick={stopRecording} style={{ background: 'var(--bg-deep)', border: '1px solid var(--red)', color: 'var(--red)' }}>
                              ⏹️ Stop Recording
                            </button>
                          )}
                          <label className="detect-btn" style={{ background: 'var(--bg-card2)', border: '1px solid var(--border)', flex: 1, cursor: 'pointer' }}>
                            📁 Upload .wav
                            <input type="file" accept="audio/wav" style={{ display: 'none' }} onChange={(e) => setDetectFile(e.target.files[0])} />
                          </label>
                        </div>
                        {detectFile && <p style={{ fontSize: '0.8rem', color: 'var(--green)' }}>✅ File ready: {detectFile.name}</p>}
                      </div>
                    )}

                    {selected === 'image' && (
                      <div className="image-detect-area">
                        <div className="webcam-container" style={{ display: webcamActive ? 'block' : 'none', marginBottom: 12 }}>
                          <video ref={videoRef} className="webcam-video" autoPlay playsInline muted />
                          <div className="webcam-overlay" />
                        </div>
                        <canvas ref={canvasRef} style={{ display: 'none' }} />

                        <div style={{ display: 'flex', gap: 12 }}>
                          <button className="detect-btn" onClick={toggleWebcam} style={{ background: webcamActive ? 'var(--red)' : 'var(--purple)' }}>
                            {webcamActive ? '🚫 Close Camera' : '📸 Open Webcam'}
                          </button>
                          <label className="detect-btn" style={{ background: 'var(--bg-card2)', border: '1px solid var(--border)', flex: 1, cursor: 'pointer' }}>
                            📁 Upload Image
                            <input type="file" accept="image/*" style={{ display: 'none' }} onChange={(e) => { setDetectFile(e.target.files[0]); setWebcamActive(false); }} />
                          </label>
                        </div>
                        {!webcamActive && detectFile && <p style={{ fontSize: '0.8rem', color: 'var(--green)', marginTop: 8 }}>✅ Image ready: {detectFile.name}</p>}
                      </div>
                    )}

                    <button
                      className="detect-btn"
                      onClick={handleDetect}
                      disabled={detecting || (!detectText && selected === 'text') || (!detectFile && !webcamActive && (selected === 'voice' || selected === 'image'))}
                      style={{ background: `var(--${selected}-mod)`, height: 56, fontSize: '1rem' }}
                    >
                      {detecting ? <><span className="spinner" /> Analyzing...</> : `Analyze ${MODALITY_MAP[selected].label}`}
                    </button>
                  </div>

                  {detectResult && (
                    <DetectResultCard result={detectResult} selected={selected} />
                  )}
                </div>
              )}
            </div>
          )}

          {/* Error box */}
          {showError && (
            <div className="panel">
              <p className="panel-title">Error</p>
              <div className="error-box">
                <span className="error-icon">⚠️</span>
                <div>
                  <p className="error-msg">{error.message}</p>
                  {error.traceback && (
                    <pre className="error-trace">{error.traceback}</pre>
                  )}
                  <button
                    onClick={handleRun}
                    style={{
                      marginTop: 12, padding: '7px 16px',
                      background: 'rgba(248,113,113,0.15)',
                      border: '1px solid rgba(248,113,113,0.4)',
                      borderRadius: 8, color: 'var(--red)',
                      cursor: 'pointer', fontSize: '0.8rem', fontWeight: 600,
                      fontFamily: 'Inter, sans-serif'
                    }}
                  >
                    ↺ Retry
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Results */}
          {showResults && (
            <ResultsPanel result={result} modality={selected} />
          )}
        </section>
      </main>
    </div>
  )
}

/* ── Tiny key-value row ───────────────────────────────────── */
function Row({ label, value }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12 }}>
      <span style={{ color: 'var(--text-muted)', flexShrink: 0, fontSize: '0.75rem', paddingTop: 1 }}>{label}</span>
      <span style={{ textAlign: 'right', color: 'var(--text-secondary)', fontSize: '0.78rem' }}>{value}</span>
    </div>
  )
}

/* ── Detection Result Card ────────────────────────────────── */
const EMOTION_EMOJI = {
  happy: '😊', sad: '😢', angry: '😠', fear: '😨', surprise: '😲',
  disgust: '🤢', neutral: '😐', pleasant_surprise: '😄', Pleasant_surprise: '😄'
}

function DetectResultCard({ result, selected }) {
  if (!result) return null
  const emotion = typeof result === 'string' ? result : result.emotion
  const confidence = result.confidence
  const allScores = result.all_scores || {}
  const faceInfo = result.face_info

  const emoji = EMOTION_EMOJI[emotion?.toLowerCase()] || '🎭'
  const sortedScores = Object.entries(allScores).sort((a, b) => b[1] - a[1])

  return (
    <div className="detect-result-card" style={{ padding: '20px 24px' }}>
      <p className="detect-result-label">Predicted Emotion</p>

      {/* Main emotion display */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 8 }}>
        <span style={{ fontSize: '2.8rem', lineHeight: 1 }}>{emoji}</span>
        <div>
          <p className="detect-result-value"
            style={{
              background: `linear-gradient(135deg, var(--${selected}-mod, #a78bfa), var(--cyan))`,
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
              fontSize: '2rem', marginBottom: 2
            }}>
            {emotion}
          </p>
          {confidence != null && (
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Confidence: <strong style={{ color: 'var(--green)' }}>{confidence.toFixed(1)}%</strong>
            </p>
          )}
        </div>
      </div>

      {/* Face detection info for image */}
      {faceInfo && (
        <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: 10, fontStyle: 'italic' }}>
          📍 {faceInfo}
        </p>
      )}

      {/* Per-emotion bar chart */}
      {sortedScores.length > 0 && (
        <div style={{ marginTop: 10 }}>
          <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: 6, letterSpacing: '0.05em', textTransform: 'uppercase' }}>All Emotions</p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
            {sortedScores.map(([label, score]) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', width: 80, flexShrink: 0, textAlign: 'right' }}>{label}</span>
                <div style={{ flex: 1, background: 'rgba(255,255,255,0.06)', borderRadius: 4, height: 8, overflow: 'hidden' }}>
                  <div style={{
                    width: `${score}%`, height: '100%', borderRadius: 4,
                    background: label === emotion
                      ? `var(--${selected}-mod, #a78bfa)`
                      : 'rgba(255,255,255,0.15)',
                    transition: 'width 0.6s ease'
                  }} />
                </div>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', width: 36, textAlign: 'right' }}>{score}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
