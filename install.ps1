# AI Facial Intelligence System - Automated Installer
# This script sets up the Python backend and React frontend environments.

$ErrorActionPreference = "Stop"

function Write-Header($msg) {
    Write-Host "`n=======================================================" -ForegroundColor Cyan
    Write-Host " $msg" -ForegroundColor White -BackgroundColor Blue
    Write-Host "=======================================================" -ForegroundColor Cyan
}

function Write-Success($msg) {
    Write-Host "[OK] $msg" -ForegroundColor Green
}

function Write-Info($msg) {
    Write-Host "[INFO] $msg" -ForegroundColor Yellow
}

function Write-ErrorMsg($msg) {
    Write-Host "[ERROR] $msg" -ForegroundColor Red
}

try {
    Write-Header "Starting Automated Installation"

    # 1. System Requirements Check
    Write-Info "Checking Prerequisites..."
    
    # Check Python
    $pythonCmd = "python"
    if (-Not (Get-Command "python" -ErrorAction SilentlyContinue)) {
        if (Get-Command "py" -ErrorAction SilentlyContinue) {
            $pythonCmd = "py"
            Write-Success "Python launcher (py) found."
        } else {
            Write-ErrorMsg "Python not found. Please install Python 3.10+."
            exit 1
        }
    }
    
    $pythonVersion = & $pythonCmd --version
    Write-Success "Using Python: $pythonVersion"

    # Check Node.js
    if (Get-Command "node" -ErrorAction SilentlyContinue) {
        $nodeVersion = node --version
        Write-Success "Node.js found: $nodeVersion"
    } else {
        Write-ErrorMsg "Node.js not found. Please install Node.js 18+."
        exit 1
    }

    # 2. Setup Python Virtual Environment
    Write-Header "Configuring Backend Environment"
    
    if (-Not (Test-Path ".venv")) {
        Write-Info "Creating virtual environment..."
        & $pythonCmd -m venv .venv
        Write-Success "Virtual environment created."
    } else {
        Write-Info "Virtual environment already exists."
    }

    Write-Info "Installing Backend Dependencies (this may take a few minutes)..."
    & ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
    & ".\.venv\Scripts\python.exe" -m pip install -r backend/requirements.txt
    Write-Success "Backend dependencies installed."

    # 3. Setup Frontend
    Write-Header "Configuring Frontend Environment"
    
    if (Test-Path "frontend") {
        Push-Location frontend
        Write-Info "Installing Frontend Dependencies..."
        npm install
        Pop-Location
        Write-Success "Frontend dependencies installed."
    } else {
        Write-ErrorMsg "Frontend directory not found!"
    }

    # 4. Verify AI Engines
    Write-Header "Verifying AI Engines"
    
    Write-Info "Checking DeepFace model availability..."
    & ".\.venv\Scripts\python.exe" -c "from deepface import DeepFace; print('[OK] DeepFace available')" | Out-Host

    Write-Header "Installation Complete!"
    Write-Success "The AI Facial Intelligence System is ready to go."
    Write-Host ""
    Write-Host "To start the system, run:" -ForegroundColor White
    Write-Host "  .\start_servers.bat" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "The dashboard will be available at: http://localhost:5173" -ForegroundColor Yellow
    Write-Host ""
    
} catch {
    Write-ErrorMsg "Installation failed: $($_.Exception.Message)"
    exit 1
}
