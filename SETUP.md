# UniMap Setup Guide

## Quick Start

### Option 1: Use the Development Script (Recommended)

1. Make sure you have Python 3.7+ and Node.js 16+ installed
2. Run the development script:
   ```bash
   ./start-dev.sh
   ```
3. Open your browser to `http://localhost:3000`

### Option 2: Manual Setup

#### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

#### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## Testing the Application

1. Open `http://localhost:3000` in your browser
2. Select a start room (e.g., "1001")
3. Select an end room (e.g., "1006")
4. Click "Find Route" to see the navigation path

## API Testing

You can test the backend API directly:
```bash
curl http://localhost:5000/1001/1006
```

## Troubleshooting

### Backend Issues
- Make sure Python virtual environment is activated
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Ensure port 5000 is not in use

### Frontend Issues
- Make sure Node.js dependencies are installed: `npm install`
- Check that port 3000 is not in use
- Clear browser cache if you see stale content

### CORS Issues
- The backend is configured to allow all origins for development
- If you see CORS errors, make sure the backend is running on port 5000

## Project Structure

```
UniMap/
├── backend/          # Flask API server
├── frontend/         # Next.js React app
├── start-dev.sh      # Development startup script
├── README.md         # Project documentation
└── SETUP.md          # This file
``` 