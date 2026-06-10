import { useState } from 'react'
import '../App.css'
import Button from './Button'
import logo from '../assets/logo.png'

function Login({ onLogin }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')

  const VALID_USERNAME = 'Theja@Mca'
  const VALID_PASSWORD = '180403'

  const handleSubmit = (e) => {
    e.preventDefault()
    if (username === VALID_USERNAME && password === VALID_PASSWORD) {
      onLogin()
    } else {
      setError('Invalid credentials. Please try again.')
    }
  }

  return (
    <div className="login-container">
      <div className="login-card glass-card animate-fade-in">
        <div className="login-header">
          <div className="logo-icon large">
            <img src={logo} alt="AI Face Intel Logo" className="login-logo" />
          </div>
          <h1 className="text-gradient">Admin Login</h1>
          <p className="subtitle">AI Facial Intelligence System</p>
        </div>

        <form onSubmit={handleSubmit} className="login-form">
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input
              type="text"
              id="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter your username"
              required
              autoComplete="username"
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              required
              autoComplete="current-password"
            />
          </div>

          {error && <div className="error-message animate-pulse">{error}</div>}

          <Button
            type="submit"
            variant="primary"
            className="btn-block"
            size="lg"
          >
            Login
          </Button>
        </form>

        <div className="login-footer">
          <p>Restricted to authorized personnel only.</p>
        </div>
      </div>

      <style jsx>{`
        .login-container {
          display: flex;
          justify-content: center;
          align-items: center;
          min-height: 100vh;
          width: 100%;
          padding: 20px;
        }

        .login-card {
          width: 100%;
          max-width: 400px;
          padding: var(--spacing-xl);
          text-align: center;
        }

        .login-header {
          margin-bottom: var(--spacing-xl);
        }

        .logo-icon.large {
          margin-bottom: var(--spacing-md);
        }

        .login-logo {
          width: 150px;
          height: 150px;
          object-fit: cover;
          border-radius: 50%;
          filter: drop-shadow(0 0 15px rgba(99, 102, 241, 0.5));
          border: 2px solid rgba(255, 255, 255, 0.2);
        }

        .login-form {
          text-align: left;
        }

        .form-group {
          margin-bottom: var(--spacing-lg);
        }

        .form-group label {
          display: block;
          margin-bottom: var(--spacing-xs);
          color: var(--text-secondary);
          font-size: 0.9rem;
          font-weight: 500;
        }

        input {
          width: 100%;
          padding: 12px 16px;
          background: var(--bg-tertiary);
          border: 1px solid var(--border-color);
          border-radius: var(--radius-md);
          color: var(--text-primary);
          font-family: inherit;
          font-size: 1rem;
          transition: all var(--transition-base);
        }

        input:focus {
          outline: none;
          border-color: var(--primary);
          box-shadow: 0 0 0 2px hsla(240, 100%, 65%, 0.2);
          background: var(--bg-secondary);
        }

        .btn-block {
          width: 100%;
          margin-top: var(--spacing-md);
        }

        .error-message {
          color: var(--danger);
          font-size: 0.85rem;
          margin-bottom: var(--spacing-md);
          text-align: center;
          background: hsla(0, 84%, 60%, 0.1);
          padding: 8px;
          border-radius: var(--radius-sm);
        }

        .login-footer {
          margin-top: var(--spacing-xl);
          color: var(--text-muted);
          font-size: 0.8rem;
        }

        .subtitle {
          color: var(--text-secondary);
          font-size: 0.9rem;
        }
      `}</style>
    </div>
  )
}

export default Login
