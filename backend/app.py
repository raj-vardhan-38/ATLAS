from flask import Flask, request, render_template, jsonify, send_file, flash, redirect, url_for, send_from_directory
import os
import uuid
from werkzeug.utils import secure_filename
from pathlib import Path
import json
from predict import run_analysis
import threading
import time

app = Flask(__name__, 
            static_folder='../frontend',
            static_url_path='/static')
app.secret_key = 'atlas_secret_key_2024'  # Change this in production

# Configuration
UPLOAD_FOLDER = Path(__file__).parent / 'temp_uploads'
REPORTS_FOLDER = Path(__file__).parent / 'reports'
ALLOWED_EXTENSIONS = {'fasta', 'fa', 'fas', 'fna'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit

# Ensure directories exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
REPORTS_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Store analysis results in memory (in production, use a database)
analysis_results = {}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_analysis_async(file_path, analysis_id):
    """Run analysis in background thread."""
    try:
        print(f"Starting analysis for {analysis_id}")
        result = run_analysis(file_path)
        analysis_results[analysis_id] = {
            'status': 'completed',
            'result': result,
            'timestamp': time.time()
        }
        print(f"Analysis completed for {analysis_id}")
    except Exception as e:
        analysis_results[analysis_id] = {
            'status': 'error',
            'error': str(e),
            'timestamp': time.time()
        }
        print(f"Analysis failed for {analysis_id}: {e}")

@app.route('/')
def index():
    """Serve the frontend home page."""
    return send_from_directory('../frontend', 'index.html')

@app.route('/analyze')
def analyze():
    """Analysis upload page."""
    return render_template('index.html')

# Additional static file routes for frontend assets
@app.route('/css/<path:filename>')
def frontend_css(filename):
    """Serve CSS files from frontend."""
    return send_from_directory('../frontend/css', filename)

@app.route('/js/<path:filename>')
def frontend_js(filename):
    """Serve JS files from frontend."""
    return send_from_directory('../frontend/js', filename)

@app.route('/assets/<path:filename>')
def frontend_assets(filename):
    """Serve asset files from frontend."""
    return send_from_directory('../frontend/assets', filename)

@app.route('/font/<path:filename>')
def frontend_fonts(filename):
    """Serve font files from frontend."""
    return send_from_directory('../frontend/font', filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start analysis."""
    try:
        print(f"Upload request received. Files: {list(request.files.keys())}")
        
        if 'file' not in request.files:
            print("No 'file' key in request.files")
            flash('No file selected')
            return redirect(url_for('index'))
        
        file = request.files['file']
        print(f"File object: {file}, filename: {file.filename}")
        
        if file.filename == '':
            print("Empty filename")
            flash('No file selected')
            return redirect(url_for('index'))
        
        if file and allowed_file(file.filename):
            # Generate unique filename and analysis ID
            filename = secure_filename(file.filename)
            analysis_id = str(uuid.uuid4())
            file_path = UPLOAD_FOLDER / f"{analysis_id}_{filename}"
            
            print(f"Saving file to: {file_path}")
            
            # Save uploaded file
            file.save(str(file_path))
            
            print(f"File saved successfully. Size: {file_path.stat().st_size} bytes")
            
            # Initialize analysis status
            analysis_results[analysis_id] = {
                'status': 'processing',
                'filename': filename,
                'timestamp': time.time()
            }
            
            # Start analysis in background thread
            thread = threading.Thread(
                target=run_analysis_async, 
                args=(str(file_path), analysis_id)
            )
            thread.daemon = True
            thread.start()
            
            print(f"Analysis started for {analysis_id}")
            return redirect(url_for('results', analysis_id=analysis_id))
        
        print(f"File type not allowed: {file.filename}")
        flash('Invalid file type. Please upload a FASTA file (.fasta, .fa, .fas, .fna)')
        return redirect(url_for('index'))
        
    except Exception as e:
        print(f"Upload error: {e}")
        flash(f'Upload failed: {str(e)}')
        return redirect(url_for('index'))

@app.route('/results/<analysis_id>')
def results(analysis_id):
    """Display analysis results."""
    if analysis_id not in analysis_results:
        flash('Analysis not found')
        return redirect(url_for('index'))
    
    return render_template('results.html', analysis_id=analysis_id)

@app.route('/api/status/<analysis_id>')
def get_status(analysis_id):
    """API endpoint to check analysis status."""
    if analysis_id not in analysis_results:
        return jsonify({'error': 'Analysis not found'}), 404
    
    status_data = analysis_results[analysis_id]
    
    if status_data['status'] == 'completed' and 'result' in status_data:
        result = status_data['result']
        
        # Prepare data for visualization
        response_data = {
            'status': 'completed',
            'filename': status_data.get('filename', 'Unknown'),
            'total_sequences': result.get('total_sequences', 0),
            'classified': result.get('classified', 0),
            'unclassified': result.get('unclassified', 0),
            'classified_results': result.get('classified_results', {}),
            'report_path': result.get('report_path', ''),
            'explorer_report': result.get('explorer_report', '')
        }
        
        return jsonify(response_data)
    
    elif status_data['status'] == 'error':
        return jsonify({
            'status': 'error',
            'error': status_data.get('error', 'Unknown error occurred')
        })
    
    else:
        return jsonify({
            'status': 'processing',
            'filename': status_data.get('filename', 'Unknown')
        })

@app.route('/download/<analysis_id>')
def download_report(analysis_id):
    """Download the generated report file."""
    if analysis_id not in analysis_results:
        flash('Analysis not found')
        return redirect(url_for('index'))
    
    status_data = analysis_results[analysis_id]
    
    if status_data['status'] != 'completed' or 'result' not in status_data:
        flash('Analysis not completed or failed')
        return redirect(url_for('results', analysis_id=analysis_id))
    
    report_path = status_data['result'].get('report_path')
    
    if not report_path or not Path(report_path).exists():
        flash('Report file not found')
        return redirect(url_for('results', analysis_id=analysis_id))
    
    return send_file(
        report_path,
        as_attachment=True,
        download_name=f"ATLAS_Report_{analysis_id[:8]}.txt",
        mimetype='text/plain'
    )

@app.route('/api/chart-data/<analysis_id>')
def get_chart_data(analysis_id):
    """API endpoint to get chart data for visualization."""
    if analysis_id not in analysis_results:
        return jsonify({'error': 'Analysis not found'}), 404
    
    status_data = analysis_results[analysis_id]
    
    if status_data['status'] != 'completed' or 'result' not in status_data:
        return jsonify({'error': 'Analysis not completed'}), 400
    
    result = status_data['result']
    classified_results = result.get('classified_results', {})
    
    # Prepare data for charts
    chart_data = {
        'pie_chart': {
            'labels': list(classified_results.keys()) + (['Unclassified'] if result.get('unclassified', 0) > 0 else []),
            'data': list(classified_results.values()) + ([result.get('unclassified', 0)] if result.get('unclassified', 0) > 0 else [])
        },
        'bar_chart': {
            'labels': list(classified_results.keys()),
            'data': list(classified_results.values())
        },
        'summary': {
            'total': result.get('total_sequences', 0),
            'classified': result.get('classified', 0),
            'unclassified': result.get('unclassified', 0),
            'classification_rate': round((result.get('classified', 0) / max(result.get('total_sequences', 1), 1)) * 100, 1)
        }
    }
    
    return jsonify(chart_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)
