import { useState, useRef, useEffect } from 'react'
import './ControlPanel.css'
import Button from './Button'

const FEATURE_LIST = [
    { id: 'face_detection', label: 'Face Detection', icon: '👤', category: 'core' },
    { id: 'face_landmarks', label: 'Face Landmarks', icon: '📍', category: 'core' },
    { id: 'face_recognition', label: 'Face Recognition', icon: '🔍', category: 'core' },
    { id: 'head_pose', label: 'Head Pose', icon: '🔄', category: 'core' },
    { id: 'eye_gaze', label: 'Eye Gaze', icon: '👁️', category: 'core' },
    { id: 'blink_detection', label: 'Blink Detection', icon: '😑', category: 'core' },
    { id: 'age_gender', label: 'Age & Gender', icon: '👥', category: 'analysis' },
    { id: 'emotion', label: 'Emotion', icon: '😊', category: 'analysis' },
    { id: 'attention', label: 'Attention Level', icon: '🎯', category: 'analysis' },
    { id: 'fatigue', label: 'Fatigue Detection', icon: '😴', category: 'analysis' },
    { id: 'stress', label: 'Stress Level', icon: '😰', category: 'analysis' },
    { id: 'anti_spoofing', label: 'Anti-Spoofing (AI)', icon: '�️', category: 'security' },
    { id: 'multi_face_tracking', label: 'Multi-Face Tracking', icon: '👥', category: 'security' },
    { id: 'background_removal', label: 'Background Removal', icon: '🖼️', category: 'effects' },
    { id: 'style_filter', label: 'Style Filters', icon: '🎨', category: 'effects' },
]

// Feature Dependencies (parent features required)
const FEATURE_DEPENDENCIES = {
    face_landmarks: ['face_detection'],
    face_recognition: ['face_detection'],
    age_gender: ['face_detection'],
    emotion: ['face_detection'],
    head_pose: ['face_landmarks'],
    eye_gaze: ['face_landmarks'],
    blink_detection: ['face_landmarks'],
    attention: ['face_landmarks', 'eye_gaze'],
    fatigue: ['face_landmarks', 'blink_detection'],
    stress: ['face_landmarks', 'emotion'],
    anti_spoofing: ['face_landmarks', 'blink_detection'],
    multi_face_tracking: ['face_detection'],
    background_removal: ['face_detection'],
}

function ControlPanel({ features, attributes, onToggleFeature, cameraActive, onStartCamera, onStopCamera, onExportReport, currentFrame }) {
    const [activeCategory, setActiveCategory] = useState('all')
    const [showRegisterModal, setShowRegisterModal] = useState(false)
    const [registerMode, setRegisterMode] = useState('idle') // 'idle', 'capturing', 'input'
    const [captureProgress, setCaptureProgress] = useState({ current: 0, target: 26 })
    const [registerData, setRegisterData] = useState({ name: '', gender: 'Male' })
    const [uploadStatus, setUploadStatus] = useState('')
    const captureIntervalRef = useRef(null)

    // Delete Person state
    const [showDeleteModal, setShowDeleteModal] = useState(false)
    const [registeredPersons, setRegisteredPersons] = useState([])
    const [selectedPerson, setSelectedPerson] = useState('')
    const [confirmDelete, setConfirmDelete] = useState(false)  // inline confirm flag
    const [deleteStatus, setDeleteStatus] = useState('')

    // Rescan state
    const [rescanStatus, setRescanStatus] = useState('')

    // AR Filter state
    const [currentARFilter, setCurrentARFilter] = useState('none')

    // Rescan dataset handler
    const handleRescanDataset = async () => {
        setRescanStatus('Scanning...')
        try {
            const response = await fetch('http://localhost:8000/recognition/rescan', { method: 'POST' })
            const result = await response.json()
            if (result.success) {
                const stats = result.stats
                setRescanStatus(`✅ ${stats.unique_persons} person(s), ${stats.total_embeddings} embeddings loaded`)
                await loadRegisteredPersons()
            } else {
                setRescanStatus(`❌ ${result.error || 'Rescan failed'}`)
            }
        } catch (error) {
            setRescanStatus('❌ Could not reach backend')
        }
        setTimeout(() => setRescanStatus(''), 4000)
    }

    // AR Filter handler
    const handleARFilterChange = async (filterName) => {
        try {
            const formData = new FormData()
            formData.append('filter_name', filterName)

            const response = await fetch('http://localhost:8000/ar-filter/set', {
                method: 'POST',
                body: formData
            })

            if (response.ok) {
                setCurrentARFilter(filterName)
                console.log(`AR Filter set to: ${filterName}`)
            }
        } catch (error) {
            console.error('Failed to set AR filter:', error)
        }
    }

    const categories = {
        all: 'All Features',
        core: 'Core',
        analysis: 'Analysis',
        security: 'Security',
        effects: 'Effects'
    }

    const filteredFeatures = activeCategory === 'all'
        ? FEATURE_LIST
        : FEATURE_LIST.filter(f => f.category === activeCategory)

    // Check if a feature can be enabled based on dependencies
    const canEnableFeature = (featureId) => {
        const deps = FEATURE_DEPENDENCIES[featureId] || []
        return deps.every(dep => features[dep] === true)
    }

    // Check if a feature is disabled (grayed out)
    const isFeatureDisabled = (featureId) => {
        return !canEnableFeature(featureId) && !features[featureId]
    }

    // Enhanced toggle handler with dependency validation
    const handleToggleFeature = (featureId) => {
        onToggleFeature(featureId, !features[featureId])
    }

    const handleStartLiveRegistration = async () => {
        try {
            // Start backend session
            const response = await fetch('http://localhost:8000/recognition/live/start', {
                method: 'POST'
            })
            const result = await response.json()

            if (result.success) {
                setShowRegisterModal(true)
                setRegisterMode('capturing')
                setCaptureProgress({ current: 0, target: result.target_samples })
                setUploadStatus('Starting capture...')

                // Ensure camera is running
                if (!cameraActive) {
                    await onStartCamera()
                    await new Promise(resolve => setTimeout(resolve, 1000))
                }

                // Start auto-capture
                startAutoCapture()
            }
        } catch (error) {
            console.error('Failed to start registration:', error)
            setUploadStatus('❌ Failed to start')
        }
    }

    const startAutoCapture = () => {
        let captureCount = 0
        let capturing = true

        captureIntervalRef.current = setInterval(async () => {
            if (!capturing) return

            if (!currentFrame) {
                // No frame yet - wait
                return
            }

            if (captureCount >= 26) {
                capturing = false
                handleCaptureComplete()
                return
            }

            try {
                // currentFrame may be a raw base64 string or a full data URI
                const dataUri = currentFrame.startsWith('data:')
                    ? currentFrame
                    : `data:image/jpeg;base64,${currentFrame}`

                const blob = await fetch(dataUri).then(r => r.blob())

                if (blob.size === 0) {
                    console.warn('Empty frame blob, skipping')
                    return
                }

                const formData = new FormData()
                formData.append('file', blob, 'capture.jpg')

                const response = await fetch('http://localhost:8000/recognition/live/capture', {
                    method: 'POST',
                    body: formData
                })

                if (!response.ok) {
                    const errJson = await response.json().catch(() => ({}))
                    console.error('Capture rejected:', errJson)
                    return
                }

                const result = await response.json()

                if (result.success) {
                    captureCount = result.samples_collected
                    setCaptureProgress({ current: captureCount, target: result.target_samples })
                    setUploadStatus(`Capturing... ${captureCount}/${result.target_samples}`)

                    if (result.complete) {
                        capturing = false
                        handleCaptureComplete()
                    }
                } else {
                    console.warn('Capture response !success:', result)
                }
            } catch (error) {
                console.error('Capture error:', error)
            }
        }, 200) // 200 ms between captures — backend parallel extraction keeps up easily
    }

    const handleCaptureComplete = () => {
        if (captureIntervalRef.current) {
            clearInterval(captureIntervalRef.current)
            captureIntervalRef.current = null
        }

        setRegisterMode('input')
        setUploadStatus('✅ Capture complete! Enter details below:')
    }

    const handleCompleteRegistration = async (e) => {
        e.preventDefault()

        const trimmedName = (registerData.name || '').trim()
        if (!trimmedName) {
            alert('Please enter a name')
            return
        }

        setUploadStatus('Finalizing registration...')

        try {
            const formData = new FormData()
            formData.append('name', trimmedName)
            formData.append('gender', registerData.gender)

            const response = await fetch('http://localhost:8000/recognition/live/complete', {
                method: 'POST',
                body: formData
            })

            const result = await response.json()

            if (response.ok && result.success) {
                setUploadStatus(`✅ ${result.message}`)
                // Refresh the persons list so the new entry shows up in Delete dropdown
                await loadRegisteredPersons()
                setTimeout(() => {
                    setShowRegisterModal(false)
                    setRegisterMode('idle')
                    setRegisterData({ name: '', gender: 'Male' })
                    setUploadStatus('')
                    setCaptureProgress({ current: 0, target: 26 })
                }, 2000)
            } else {
                const errorMsg = result.error || result.message || 'Registration failed'
                setUploadStatus(`❌ ${errorMsg}`)
                console.error('Registration error:', errorMsg)
            }
        } catch (error) {
            console.error('Registration request failed:', error)
            setUploadStatus('❌ Connection Error — is the backend running?')
        }
    }

    const handleCancelRegistration = async () => {
        if (captureIntervalRef.current) {
            clearInterval(captureIntervalRef.current)
            captureIntervalRef.current = null
        }

        try {
            await fetch('http://localhost:8000/recognition/live/cancel', { method: 'POST' })
        } catch (error) {
            // Ignore — session may have already ended
            console.warn('Cancel endpoint error (ignored):', error)
        }

        setShowRegisterModal(false)
        setRegisterMode('idle')
        setRegisterData({ name: '', gender: 'Male' })
        setUploadStatus('')
        setCaptureProgress({ current: 0, target: 26 })
    }

    const loadRegisteredPersons = async () => {
        try {
            const response = await fetch('http://localhost:8000/recognition/persons')
            const data = await response.json()
            if (data.success) {
                setRegisteredPersons(data.persons)
            }
        } catch (error) {
            console.error('Failed to load persons:', error)
        }
    }

    const handleDeletePerson = async () => {
        if (!selectedPerson) {
            setDeleteStatus('⚠️ Please select a person first.')
            return
        }

        if (!confirmDelete) {
            setConfirmDelete(true)
            return
        }

        // Second click: actually delete
        setDeleteStatus('Deleting...')
        setConfirmDelete(false)
        try {
            const formData = new FormData()
            formData.append('name', selectedPerson)

            const response = await fetch('http://localhost:8000/recognition/delete', {
                method: 'POST',
                body: formData
            })
            const result = await response.json()

            if (result.success) {
                setDeleteStatus(`✅ Deleted ${selectedPerson} (${result.removed_count} embeddings removed)`)
                setSelectedPerson('')
                // Immediately reload the list so UI reflects the change
                await loadRegisteredPersons()
                setTimeout(() => {
                    setShowDeleteModal(false)
                    setDeleteStatus('')
                }, 2000)
            } else {
                setDeleteStatus(`❌ ${result.error || 'Failed to delete person'}`)
            }
        } catch (error) {
            console.error('Delete failed:', error)
            setDeleteStatus('❌ Failed to connect to server. Is the backend running?')
        }
    }



    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (captureIntervalRef.current) {
                clearInterval(captureIntervalRef.current)
            }
        }
    }, [])

    return (
        <div className="control-panel">
            <div className="panel-header">
                <h2>🎛️ Control Panel</h2>
            </div>

            {/* Multi-Face Tracking Stats */}
            <div className="tracking-stats">
                <div className="stat-item">
                    <span className="stat-label">👥 Faces Detected:</span>
                    <span className="stat-value">{attributes.face_count || 0}</span>
                </div>
                <div className="stat-item">
                    <span className="stat-label">🔒 Anti-Spoofing:</span>
                    <span className="stat-value" style={{ color: '#00ff00' }}>Active</span>
                </div>
                <div className="stat-item">
                    <span className="stat-label">⚡ FPS:</span>
                    <span className="stat-value">{attributes.fps || 0}</span>
                </div>
            </div>

            {/* Camera Controls */}
            <div className="control-section">
                <h3 className="section-title">Camera</h3>
                <div className="button-group">
                    {!cameraActive ? (
                        <Button variant="success" onClick={onStartCamera} icon="▶️">
                            Start Camera
                        </Button>
                    ) : (
                        <Button variant="danger" onClick={onStopCamera} icon="⏹️">
                            Stop Camera
                        </Button>
                    )}
                </div>
            </div>

            {/* Data Management */}
            <div className="control-section">
                <h3 className="section-title">Data Management</h3>
                <div className="button-group">
                    <Button
                        variant="primary"
                        onClick={handleStartLiveRegistration}
                        disabled={registerMode !== 'idle'}
                        icon="📸"
                    >
                        Register Person (Live)
                    </Button>

                    <Button
                        variant="danger"
                        onClick={() => {
                            loadRegisteredPersons()
                            setShowDeleteModal(true)
                        }}
                        icon="🗑️"
                    >
                        Delete Person
                    </Button>

                    <Button
                        variant="secondary"
                        onClick={handleRescanDataset}
                        disabled={rescanStatus === 'Scanning...'}
                        loading={rescanStatus === 'Scanning...'}
                        icon="🔄"
                        title="Re-scan dataset folder and reload all embeddings (no restart needed)"
                    >
                        Rescan Dataset
                    </Button>

                    {rescanStatus && (
                        <div style={{ fontSize: '12px', marginTop: '6px', color: rescanStatus.startsWith('✅') ? '#4caf50' : rescanStatus.startsWith('❌') ? '#f44336' : '#aaa' }}>
                            {rescanStatus}
                        </div>
                    )}
                </div>
            </div>

            {/* Live Registration Modal */}
            {showRegisterModal && (
                <div className="modal-overlay">
                    <div className="modal-content glass-card">
                        <h3>🎥 Live Registration</h3>

                        {registerMode === 'capturing' && (
                            <div className="capture-progress">
                                <div className="progress-bar-container">
                                    <div
                                        className="progress-bar-fill"
                                        style={{ width: `${(captureProgress.current / captureProgress.target) * 100}%` }}
                                    ></div>
                                </div>
                                <p className="progress-text">
                                    {captureProgress.current} / {captureProgress.target} samples captured
                                </p>
                                <p className="status-text">{uploadStatus}</p>
                            </div>
                        )}

                        {registerMode === 'input' && (
                            <form onSubmit={handleCompleteRegistration}>
                                <div className="form-group">
                                    <label>Name:</label>
                                    <input
                                        type="text"
                                        placeholder="Enter person's name"
                                        value={registerData.name}
                                        onChange={(e) => setRegisterData({ ...registerData, name: e.target.value })}
                                        required
                                    />
                                </div>

                                <div className="form-group">
                                    <label>Gender:</label>
                                    <select
                                        value={registerData.gender}
                                        onChange={(e) => setRegisterData({ ...registerData, gender: e.target.value })}
                                    >
                                        <option value="Male">Male</option>
                                        <option value="Female">Female</option>
                                        <option value="Other">Other</option>
                                    </select>
                                </div>

                                {uploadStatus && <p className="status-message">{uploadStatus}</p>}

                                <div className="modal-actions">
                                    <Button type="submit" variant="success">
                                        Complete Registration
                                    </Button>
                                    <Button
                                        variant="secondary"
                                        onClick={handleCancelRegistration}
                                    >
                                        Cancel
                                    </Button>
                                </div>
                            </form>
                        )}

                        {registerMode === 'capturing' && (
                            <div className="modal-actions">
                                <Button
                                    variant="danger"
                                    onClick={handleCancelRegistration}
                                >
                                    Cancel
                                </Button>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Delete Person Modal */}
            {showDeleteModal && (
                <div className="modal-overlay" onClick={(e) => {
                    // Close modal when clicking the backdrop
                    if (e.target === e.currentTarget) {
                        setShowDeleteModal(false)
                        setSelectedPerson('')
                        setConfirmDelete(false)
                        setDeleteStatus('')
                    }
                }}>
                    <div className="modal-content glass-card">
                        <h3>🗑️ Delete Registered Person</h3>

                        <div className="form-group">
                            <label>Select Person to Delete:</label>
                            <select
                                value={selectedPerson}
                                onChange={(e) => { setSelectedPerson(e.target.value); setConfirmDelete(false); setDeleteStatus('') }}
                                className="select-input"
                            >
                                <option value="">-- Select Person --</option>
                                {registeredPersons.map(person => (
                                    <option key={person.name} value={person.name}>
                                        {person.name} ({person.count} images, {person.gender})
                                    </option>
                                ))}
                            </select>
                        </div>

                        {/* Inline confirmation prompt */}
                        {confirmDelete && (
                            <div style={{
                                background: '#3a1a1a',
                                border: '1px solid #ff4444',
                                borderRadius: '8px',
                                padding: '10px 14px',
                                marginBottom: '12px',
                                color: '#ff8888',
                                fontSize: '13px'
                            }}>
                                ⚠️ Are you sure? This will permanently remove <strong>{selectedPerson}</strong> and all their images.
                                <br /><span style={{ color: '#aaa', fontSize: '12px' }}>Click &quot;Delete&quot; again to confirm.</span>
                            </div>
                        )}

                        {/* Status message */}
                        {deleteStatus && (
                            <div style={{ marginBottom: '10px', fontSize: '13px', color: deleteStatus.startsWith('✅') ? '#4caf50' : deleteStatus.startsWith('❌') ? '#f44336' : '#aaa' }}>
                                {deleteStatus}
                            </div>
                        )}

                        <div className="modal-actions">
                            <Button
                                variant="danger"
                                onClick={handleDeletePerson}
                                disabled={!selectedPerson || deleteStatus === 'Deleting...'}
                                loading={deleteStatus === 'Deleting...'}
                            >
                                {confirmDelete ? '⚠️ Confirm Delete' : 'Delete Selected Person'}
                            </Button>
                            <Button
                                variant="secondary"
                                onClick={() => {
                                    setShowDeleteModal(false)
                                    setSelectedPerson('')
                                    setConfirmDelete(false)
                                    setDeleteStatus('')
                                }}
                            >
                                Cancel
                            </Button>
                        </div>
                    </div>
                </div>
            )}

            {/* Feature Categories */}
            <div className="control-section">
                <h3 className="section-title">Feature Categories</h3>
                <div className="category-filters">
                    {Object.entries(categories).map(([key, label]) => (
                        <button
                            key={key}
                            className={`category-btn ${activeCategory === key ? 'active' : ''}`}
                            onClick={() => setActiveCategory(key)}
                        >
                            {label}
                        </button>
                    ))}
                </div>
            </div>

            {/* Feature Toggles */}
            <div className="features-section">
                <h3>Features</h3>

                {filteredFeatures.map(feature => {
                    const isEnabled = features[feature.id]
                    const isDisabled = isFeatureDisabled(feature.id)

                    return (
                        <div
                            key={feature.id}
                            className={`feature-toggle-item ${isDisabled ? 'disabled' : ''}`}
                            title={isDisabled ? 'Enable required dependencies first' : ''}
                        >
                            <div className="feature-info">
                                <span className="feature-icon">{feature.icon}</span>
                                <span className="feature-label">{feature.label}</span>
                            </div>
                            <label className="toggle-switch">
                                <input
                                    type="checkbox"
                                    checked={isEnabled}
                                    onChange={() => handleToggleFeature(feature.id)}
                                    disabled={isDisabled}
                                />
                                <span className="toggle-slider"></span>
                            </label>
                        </div>
                    )
                })}
            </div>

            {/* AR Filters */}
            <div className="control-section">
                <h3 className="section-title">🎭 AR Filters</h3>
                <select
                    value={currentARFilter}
                    onChange={(e) => handleARFilterChange(e.target.value)}
                    style={{
                        width: '100%',
                        padding: '12px',
                        fontSize: '14px',
                        borderRadius: '8px',
                        border: '2px solid #444',
                        backgroundColor: '#2a2a2a',
                        color: '#fff',
                        cursor: 'pointer',
                        marginBottom: '10px'
                    }}
                >
                    <option value="none">✨ None</option>
                    <option value="dog">🐶 Dog Ears & Tongue</option>
                    <option value="sunglasses">😎 Sunglasses</option>
                    <option value="flower_crown">🌸 Flower Crown</option>
                    <option value="big_eyes">👁️ Big Eyes</option>
                    <option value="neon_outline">✨ Neon Outline</option>
                    <option value="mask">🎭 Mask Overlay</option>
                    <option value="beauty">💄 Beauty Filter</option>
                    <option value="face_swap">🔄 Face Swap</option>
                </select>
            </div>

            {/* Export */}
            <div className="control-section">
                <h3 className="section-title">Export</h3>
                <Button variant="warning" onClick={onExportReport} icon="📄">
                    Generate Report
                </Button>
            </div>
        </div>
    )
}

export default ControlPanel
