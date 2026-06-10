import { useState, useEffect, useRef } from 'react'
import './App.css'
import VideoFeed from './components/VideoFeed'
import ControlPanel from './components/ControlPanel'
import StatsPanel from './components/StatsPanel'
import Login from './components/Login'
import Button from './components/Button'
import logo from './assets/logo.png'
import axios from 'axios'

const API_URL = 'http://localhost:8000'

function App() {
  const [cameraActive, setCameraActive] = useState(false)
  const [features, setFeatures] = useState({})
  const [attributes, setAttributes] = useState({ faces: [], fps: 0 })
  const [systemHealth, setSystemHealth] = useState('checking')
  const [currentFrame, setCurrentFrame] = useState(null)
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const wsRef = useRef(null)

  // ... (useEffects)


  useEffect(() => {
    checkHealth()
    const interval = setInterval(checkHealth, 5000)
    return () => clearInterval(interval)
  }, [])

  // Fetch features on mount
  useEffect(() => {
    fetchFeatures()
  }, [])

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API_URL}/health`)
      setSystemHealth('healthy')
      setCameraActive(response.data.camera_active)
    } catch (error) {
      setSystemHealth('error')
    }
  }

  const fetchFeatures = async () => {
    try {
      const response = await axios.get(`${API_URL}/features/list`)
      setFeatures(response.data)
    } catch (error) {
      console.error('Failed to fetch features:', error)
    }
  }

  // Voice Command Setup
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition()
      recognition.continuous = true
      recognition.interimResults = false
      recognition.lang = 'en-US'

      recognition.onresult = (event) => {
        const transcript = event.results[event.results.length - 1][0].transcript.toLowerCase().trim()
        console.log('Voice Command:', transcript)

        if (transcript.includes('start camera')) {
          startCamera()
        } else if (transcript.includes('stop camera')) {
          stopCamera()
        }
      }

      recognition.start()
      return () => recognition.stop()
    }
  }, [])

  const startCamera = async () => {
    try {
      if (cameraActive) return
      await axios.post(`${API_URL}/camera/start`, null, { params: { source: 0 } })
      setCameraActive(true)
      connectWebSocket()
    } catch (error) {
      console.error('Failed to start camera:', error)
    }
  }

  const stopCamera = async () => {
    try {
      await axios.post(`${API_URL}/camera/stop`)
      setCameraActive(false)
      if (wsRef.current) {
        wsRef.current.close()
      }
    } catch (error) {
      console.error('Failed to stop camera:', error)
    }
  }

  const connectWebSocket = () => {
    const ws = new WebSocket('ws://localhost:8000/ws/video')

    ws.onopen = () => {
      console.log('WebSocket connected')
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setAttributes(data.attributes)
      if (data.frame) {
        setCurrentFrame(`data:image/jpeg;base64,${data.frame}`)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    ws.onclose = () => {
      console.log('WebSocket disconnected')
    }

    wsRef.current = ws
  }

  const toggleFeature = async (feature, enabled) => {
    try {
      const response = await axios.post(`${API_URL}/features/toggle`, null, {
        params: { feature, enabled }
      })

      // Update with the full state from backend (handles auto-enabled dependencies)
      if (response.data.all_features) {
        setFeatures(response.data.all_features)
      } else {
        setFeatures(prev => ({ ...prev, [feature]: enabled }))
      }
    } catch (error) {
      console.error('Failed to toggle feature:', error)
    }
  }

  const handleImageUpload = async (file) => {
    const formData = new FormData()
    formData.append('file', file)

    try {
      // Stop camera if active to show uploaded image
      if (cameraActive) {
        await stopCamera()
      }

      const response = await axios.post(`${API_URL}/upload/image`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      // Display processed image and attributes
      if (response.data.success && response.data.processed_image) {
        setCurrentFrame(`data:image/jpeg;base64,${response.data.processed_image}`)
        setAttributes(response.data.attributes)
      }
    } catch (error) {
      console.error('Failed to upload image:', error)
      alert('Failed to process image. Please try again.')
    }
  }

  const handleExportReport = async () => {
    try {
      const response = await axios.post(`${API_URL}/report/export`, null, {
        responseType: 'blob'
      })

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `report-${new Date().toISOString()}.pdf`)
      document.body.appendChild(link)
      link.click()
      link.remove()
    } catch (error) {
      console.error('Failed to export report:', error)
      alert("No data available to export. Start the camera and let it run for a while.")
    }
  }

  const handleLogout = async () => {
    if (cameraActive) {
      await stopCamera()
    }
    setIsLoggedIn(false)
  }

  if (!isLoggedIn) {
    return <Login onLogin={() => setIsLoggedIn(true)} />
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header glass-card">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-icon">
              <img src={logo} alt="Logo" className="app-logo" />
            </div>
            <div>
              <h1 className="text-gradient">AI Facial Intelligence</h1>
              <p className="subtitle">Unified Real-Time Analysis System</p>
            </div>
          </div>

          <div className="header-stats">
            <div className={`status-indicator ${systemHealth === 'healthy' ? 'success' : 'danger'}`}>
              {systemHealth === 'healthy' ? '● Online' : '● Offline'}
            </div>
            <div className="fps-counter">
              <span className="fps-label">FPS:</span>
              <span className="fps-value">{attributes.fps || 0}</span>
            </div>
            <Button
              variant="danger"
              size="sm"
              onClick={handleLogout}
              icon="🚪"
              title="Logout and Exit"
            >
              Exit
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Left Sidebar - Controls */}
        <aside className="sidebar left-sidebar glass-card">
          <ControlPanel
            features={features}
            attributes={attributes}
            onToggleFeature={toggleFeature}
            cameraActive={cameraActive}
            onStartCamera={startCamera}
            onStopCamera={stopCamera}
            onImageUpload={handleImageUpload}
            onExportReport={handleExportReport}
            currentFrame={currentFrame}
          />
        </aside>

        {/* Center - Video Feed */}
        <section className="video-section">
          <VideoFeed
            frame={currentFrame}
            cameraActive={cameraActive}
          />
        </section>

        {/* Right Sidebar - Stats */}
        <aside className="sidebar right-sidebar glass-card">
          <StatsPanel attributes={attributes} />
        </aside>
      </main>
    </div>
  )
}

export default App
