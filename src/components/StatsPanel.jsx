import './StatsPanel.css'

function StatsPanel({ attributes }) {
    const { faces = [], fps = 0 } = attributes

    // Detect global spoof alert — any face that is_live === false
    const spoofDetected = faces.some(
        f => f.anti_spoof && f.anti_spoof.is_live === false
    )

    const getStatusColor = (level) => {
        if (level === 'HIGH') return 'success'
        if (level === 'MEDIUM') return 'warning'
        if (level === 'LOW') return 'danger'
        return 'secondary'
    }

    return (
        <div className="stats-panel">
            <div className="panel-header">
                <h2>Live Analysis</h2>
                <div className="badge">{faces.length} Face{faces.length !== 1 ? 's' : ''}</div>
            </div>

            {/* Global Spoof Alert Banner */}
            {spoofDetected && (
                <div style={{
                    background: 'linear-gradient(135deg, #ff0000cc, #8b0000cc)',
                    border: '2px solid #ff3333',
                    borderRadius: '10px',
                    padding: '10px 14px',
                    marginBottom: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    animation: 'spoofPulse 1s ease-in-out infinite',
                    boxShadow: '0 0 18px #ff000088'
                }}>
                    <span style={{ fontSize: '22px' }}>🚨</span>
                    <div>
                        <div style={{ color: '#fff', fontWeight: 800, fontSize: '13px', letterSpacing: '0.5px' }}>
                            SPOOFING ATTEMPT DETECTED
                        </div>
                        <div style={{ color: '#ffaaaa', fontSize: '11px' }}>
                            Liveness verification failed — potential fake face
                        </div>
                    </div>
                </div>
            )}

            {/* Performance Stats */}
            <div className="stats-section">
                <h3 className="section-title">Performance</h3>
                <div className="stat-card">
                    <div className="stat-label">Frame Rate</div>
                    <div className="stat-value">{fps} <span className="stat-unit">FPS</span></div>
                </div>
            </div>

            {/* Face Analysis */}
            {faces.length > 0 ? (
                <div className="stats-section">
                    <h3 className="section-title">Face Analysis</h3>
                    {faces.map((face, index) => (
                        <div key={index} className="face-card-container">

                            {/* SECTION 1: IDENTITY */}
                            <div className="face-card identity-section">
                                <div className="card-header-label">IDENTITY</div>
                                <div className="face-header">
                                    {face.name ? (
                                        <div className="face-identity">
                                            <div className="face-name-container">
                                                {face.name === 'Unknown' ? (
                                                    <span className="name-main-line">❓ Unknown</span>
                                                ) : (() => {
                                                    const parts = face.name.split(' ');
                                                    if (parts.length > 1) {
                                                        const firstPart = parts[0];
                                                        const remainingParts = parts.slice(1).join(' ');
                                                        return (
                                                            <>
                                                                <span className="name-first-line">👤 {firstPart}</span>
                                                                <span className="name-main-line">{remainingParts}</span>
                                                            </>
                                                        );
                                                    } else {
                                                        return <span className="name-main-line">👤 {face.name}</span>;
                                                    }
                                                })()}
                                            </div>
                                            {face.recognition_confidence && (
                                                <span
                                                    className={`recognition-badge ${face.recognition_confidence > 80 ? 'high-confidence' :
                                                        face.recognition_confidence > 60 ? 'medium-confidence' :
                                                            'low-confidence'
                                                        }`}
                                                >
                                                    {face.recognition_confidence.toFixed(1)}%
                                                </span>
                                            )}
                                        </div>
                                    ) : (
                                        <span className="face-number">Face #{index + 1}</span>
                                    )}
                                </div>
                            </div>

                            {/* SECTION 2: DEMOGRAPHICS */}
                            <div className="face-card demographics-section">
                                <div className="card-header-label">DEMOGRAPHICS</div>
                                <div className="demographics-grid">
                                    {/* Gender: Prioritize Registered over Predicted */}
                                    <div className="mini-stat">
                                        <span className="mini-label">Gender</span>
                                        <span className="mini-value highlight">
                                            {face.registered_gender && face.registered_gender !== 'Unknown'
                                                ? face.registered_gender
                                                : (face.gender || 'Analyzing...')}
                                        </span>
                                        {face.registered_gender && face.registered_gender !== 'Unknown' && (
                                            <span className="source-tag">Registry</span>
                                        )}
                                    </div>

                                    {/* Age */}
                                    <div className="mini-stat">
                                        <span className="mini-label">Age</span>
                                        <span className="mini-value big">
                                            {face.age ? face.age : '--'}
                                        </span>
                                    </div>

                                    {/* Emotion */}
                                    {face.emotion && (
                                        <div className="mini-stat">
                                            <span className="mini-label">Emotion</span>
                                            <span className="status-pill info">{face.emotion.toUpperCase()}</span>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* SECTION 3: ADVANCED METRICS */}
                            {(face.attention || face.stress || face.fatigue || face.blink) && (
                                <div className="face-card metrics-section">
                                    <div className="card-header-label">METRICS</div>

                                    {/* Attention */}
                                    {face.attention && (
                                        <div className="attribute-group">
                                            <div className="attribute-title">
                                                <span className="attr-icon">🎯</span> Attention
                                            </div>
                                            <div className={`status-pill ${getStatusColor(face.attention.level)}`}>
                                                {face.attention.level}
                                            </div>
                                        </div>
                                    )}

                                    {/* Stress */}
                                    {face.stress && (
                                        <div className="attribute-group">
                                            <div className="attribute-title">
                                                <span className="attr-icon">😰</span> Stress
                                            </div>
                                            <div className={`status-pill ${getStatusColor(face.stress.level === 'LOW' ? 'HIGH' : face.stress.level === 'HIGH' ? 'LOW' : 'MEDIUM')}`}>
                                                {face.stress.level}
                                            </div>
                                        </div>
                                    )}

                                    {/* Fatigue */}
                                    {face.fatigue && (
                                        <div className="attribute-group">
                                            <div className="attribute-title">
                                                <span className="attr-icon">😴</span> Fatigue
                                            </div>
                                            <div className={`status-pill ${getStatusColor(face.fatigue.level === 'LOW' ? 'HIGH' : face.fatigue.level === 'HIGH' ? 'LOW' : 'MEDIUM')}`}>
                                                {face.fatigue.level}
                                            </div>
                                        </div>
                                    )}

                                    {/* Blink Detection */}
                                    {face.blink && (
                                        <div className="attribute-group">
                                            <div className="attribute-title">
                                                <span className="attr-icon">👁️</span> Blink Detection
                                            </div>
                                            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', alignItems: 'flex-end' }}>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                    <span style={{ fontSize: '11px', color: '#aaa' }}>Count</span>
                                                    <span style={{
                                                        background: '#1e3a5f',
                                                        color: '#60b4ff',
                                                        fontWeight: 700,
                                                        fontSize: '14px',
                                                        padding: '2px 10px',
                                                        borderRadius: '8px',
                                                        minWidth: '36px',
                                                        textAlign: 'center'
                                                    }}>
                                                        {face.blink.count ?? 0}
                                                    </span>
                                                </div>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                    <span style={{ fontSize: '11px', color: '#aaa' }}>EAR</span>
                                                    <span style={{ fontSize: '12px', color: '#ccc' }}>
                                                        {face.blink.avg_ear?.toFixed(3) ?? '--'}
                                                    </span>
                                                </div>
                                                {face.blink.is_blinking && (
                                                    <span style={{
                                                        background: '#ff4d4d22',
                                                        color: '#ff6b6b',
                                                        fontSize: '11px',
                                                        padding: '2px 8px',
                                                        borderRadius: '6px',
                                                        fontWeight: 600,
                                                        animation: 'pulse 0.5s ease-in-out'
                                                    }}>
                                                        😑 BLINKING
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* SECTION 4: SECURITY */}
                            {face.anti_spoof && Object.keys(face.anti_spoof).length > 0 && (
                                <div className="face-card" style={{
                                    border: face.anti_spoof.is_live === false
                                        ? '2px solid #ff3333'
                                        : '2px solid #22c55e33',
                                    background: face.anti_spoof.is_live === false
                                        ? 'linear-gradient(135deg, #1a000088, #2a000088)'
                                        : 'linear-gradient(135deg, #001a0888, #00280888)'
                                }}>
                                    <div className="card-header-label" style={{
                                        color: face.anti_spoof.is_live === false ? '#ff6666' : '#22c55e'
                                    }}>SECURITY</div>
                                    <div className="attribute-group">
                                        <div className="attribute-title">
                                            <span className="attr-icon">
                                                {face.anti_spoof.is_live ? '🛡️' : '⚠️'}
                                            </span> Liveness
                                        </div>
                                        {face.anti_spoof.is_live === false ? (
                                            <div style={{
                                                display: 'flex',
                                                flexDirection: 'column',
                                                alignItems: 'flex-end',
                                                gap: '4px'
                                            }}>
                                                <span style={{
                                                    background: '#ff000033',
                                                    color: '#ff4444',
                                                    fontWeight: 800,
                                                    fontSize: '12px',
                                                    padding: '3px 10px',
                                                    borderRadius: '6px',
                                                    border: '1px solid #ff3333',
                                                    animation: 'spoofPulse 1s ease-in-out infinite'
                                                }}>
                                                    🚨 SPOOF
                                                </span>
                                                {face.anti_spoof.reason && (
                                                    <span style={{ fontSize: '10px', color: '#ff9999', textAlign: 'right', maxWidth: '120px' }}>
                                                        {face.anti_spoof.reason}
                                                    </span>
                                                )}
                                            </div>
                                        ) : (
                                            <span style={{
                                                background: '#00ff4433',
                                                color: '#22c55e',
                                                fontWeight: 700,
                                                fontSize: '12px',
                                                padding: '3px 10px',
                                                borderRadius: '6px',
                                                border: '1px solid #22c55e44'
                                            }}>✅ LIVE</span>
                                        )}
                                    </div>
                                    {face.anti_spoof.confidence !== undefined && (
                                        <div className="attribute-group" style={{ marginTop: '4px' }}>
                                            <div className="attribute-title" style={{ fontSize: '11px', color: '#aaa' }}>
                                                Confidence
                                            </div>
                                            <span style={{ fontSize: '12px', color: '#ccc' }}>
                                                {(face.anti_spoof.confidence * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                    )}
                                </div>
                            )}

                        </div>
                    ))}
                </div>
            ) : (
                <div className="empty-state">
                    <div className="empty-icon">👤</div>
                    <p>No faces detected</p>
                    <span className="empty-hint">Start the camera to begin analysis</span>
                </div>
            )}
        </div>
    )
}

export default StatsPanel
