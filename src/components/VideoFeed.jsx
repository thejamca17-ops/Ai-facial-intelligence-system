import './VideoFeed.css'

function VideoFeed({ frame, cameraActive }) {
    return (
        <div className="video-feed-container glass-card">
            <div className="video-header">
                <h2>Live Video Feed</h2>
                <div className={`recording-indicator ${cameraActive ? 'active' : ''}`}>
                    <span className="rec-dot"></span>
                    {cameraActive ? 'LIVE' : 'STOPPED'}
                </div>
            </div>

            <div className="video-wrapper">
                {frame ? (
                    <img
                        src={frame}
                        alt="Video feed"
                        className="video-frame"
                    />
                ) : (
                    <div className="video-placeholder">
                        <div className="placeholder-content">
                            <div className="camera-icon">📹</div>
                            <h3>No Video Feed</h3>
                            <p>Start the camera to begin real-time analysis</p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}

export default VideoFeed
