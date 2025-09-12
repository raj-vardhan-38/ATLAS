# ATLAS Flask Web Application

A web-based interface for the ATLAS (AI Taxonomic Learning & Analysis System) that allows users to upload FASTA files and get comprehensive biodiversity analysis results.

## Features

- **File Upload**: Drag-and-drop interface for FASTA files (.fasta, .fa, .fas, .fna)
- **Real-time Processing**: Background analysis with live status updates
- **AI Classification**: Identifies known organisms using trained ML models
- **Novel Taxa Discovery**: Explores unclassified sequences for potential new species
- **Rich Visualization**: Interactive charts and tables showing results
- **Report Generation**: Downloadable detailed analysis reports
- **Modern UI**: Responsive design with ATLAS branding

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the required model files in the `models/` directory:
   - `16s_genus_classifier.h5`
   - `16s_genus_vectorizer.pkl`
   - `16s_genus_label_encoder.pkl`
   - Additional models for 18s, coi, its markers
   - `explorer_doc2vec.model`

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Upload a FASTA file and wait for analysis results

4. View interactive charts, tables, and download detailed reports

## API Endpoints

- `GET /` - Main upload page
- `POST /upload` - Handle file upload and start analysis
- `GET /results/<analysis_id>` - View results page
- `GET /api/status/<analysis_id>` - Check analysis status (JSON)
- `GET /api/chart-data/<analysis_id>` - Get chart data (JSON)
- `GET /download/<analysis_id>` - Download analysis report

## File Structure

```
backend/
├── app.py                 # Main Flask application
├── predict.py            # Core analysis engine
├── requirements.txt      # Python dependencies
├── templates/           # HTML templates
│   ├── base.html
│   ├── index.html
│   └── results.html
├── static/              # Static assets
│   └── font/           # Font files
├── models/             # AI model files
├── temp_uploads/       # Temporary file storage
└── reports/           # Generated reports
```

## Technology Stack

- **Backend**: Flask, Python
- **Frontend**: HTML, Tailwind CSS, JavaScript
- **Charts**: Chart.js
- **AI/ML**: TensorFlow, scikit-learn, Biopython
- **Analysis**: HDBSCAN clustering, Doc2Vec, BLAST

## Configuration

The application uses the following default settings:
- Upload limit: 50MB
- Supported formats: .fasta, .fa, .fas, .fna
- Port: 5000
- Host: 0.0.0.0 (all interfaces)

## Security Notes

- Change the `app.secret_key` in production
- Consider implementing user authentication for production use
- Set appropriate file size limits based on your server capacity
- Use HTTPS in production environments
