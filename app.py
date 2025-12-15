"""
Flask Web Application for Ellipse Detection

This application provides a web interface for real-time ellipse detection
using the device's camera or uploaded images.
"""

import os
import sys
import base64
import json
from io import BytesIO
from flask import Flask, render_template, request, jsonify, Response
import numpy as np
import cv2
from PIL import Image

# Add python_src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_src'))
from ellipse_detector import EllipseDetector, Ellipse

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# Global detector instance
detector = EllipseDetector(Tac=165, Tr=0.6, specified_polarity=0)


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    """
    Detect ellipses in an uploaded image or camera frame.
    
    Accepts JSON with base64-encoded image data.
    Returns detected ellipses and processed image.
    """
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get parameters
        tac = data.get('tac', 165)
        tr = data.get('tr', 0.6)
        polarity = data.get('polarity', 0)
        
        # Update detector parameters if needed
        detector.Tac = tac
        detector.Tr = tr
        detector.specified_polarity = polarity
        
        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Detect ellipses
        ellipses, edge = detector.detect(image)
        
        # Draw ellipses on image
        result_image = detector.draw_ellipses(image, ellipses)
        
        # Encode result image to base64
        _, buffer = cv2.imencode('.jpg', result_image)
        result_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Encode edge image
        _, edge_buffer = cv2.imencode('.jpg', edge)
        edge_b64 = base64.b64encode(edge_buffer).decode('utf-8')
        
        # Format ellipse data
        ellipse_data = []
        for e in ellipses:
            ellipse_data.append({
                'center_x': round(e.center_x, 2),
                'center_y': round(e.center_y, 2),
                'semi_major': round(e.semi_major, 2),
                'semi_minor': round(e.semi_minor, 2),
                'angle_deg': round(np.degrees(e.angle), 2)
            })
        
        return jsonify({
            'success': True,
            'num_ellipses': len(ellipses),
            'ellipses': ellipse_data,
            'result_image': f'data:image/jpeg;base64,{result_b64}',
            'edge_image': f'data:image/jpeg;base64,{edge_b64}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/detect_file', methods=['POST'])
def detect_file():
    """
    Detect ellipses in an uploaded image file.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get parameters from form
        tac = float(request.form.get('tac', 165))
        tr = float(request.form.get('tr', 0.6))
        polarity = int(request.form.get('polarity', 0))
        
        # Update detector parameters
        detector.Tac = tac
        detector.Tr = tr
        detector.specified_polarity = polarity
        
        # Read and decode image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Failed to decode image file'}), 400
        
        # Detect ellipses
        ellipses, edge = detector.detect(image)
        
        # Draw ellipses on image
        result_image = detector.draw_ellipses(image, ellipses)
        
        # Encode result image to base64
        _, buffer = cv2.imencode('.jpg', result_image)
        result_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Encode edge image
        _, edge_buffer = cv2.imencode('.jpg', edge)
        edge_b64 = base64.b64encode(edge_buffer).decode('utf-8')
        
        # Format ellipse data
        ellipse_data = []
        for e in ellipses:
            ellipse_data.append({
                'center_x': round(e.center_x, 2),
                'center_y': round(e.center_y, 2),
                'semi_major': round(e.semi_major, 2),
                'semi_minor': round(e.semi_minor, 2),
                'angle_deg': round(np.degrees(e.angle), 2)
            })
        
        return jsonify({
            'success': True,
            'num_ellipses': len(ellipses),
            'ellipses': ellipse_data,
            'result_image': f'data:image/jpeg;base64,{result_b64}',
            'edge_image': f'data:image/jpeg;base64,{edge_b64}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/parameters', methods=['GET'])
def get_parameters():
    """Get current detector parameters."""
    return jsonify({
        'tac': detector.Tac,
        'tr': detector.Tr,
        'polarity': detector.specified_polarity
    })


@app.route('/api/parameters', methods=['POST'])
def set_parameters():
    """Set detector parameters."""
    try:
        data = request.get_json()
        
        if 'tac' in data:
            detector.Tac = float(data['tac'])
        if 'tr' in data:
            detector.Tr = float(data['tr'])
        if 'polarity' in data:
            detector.specified_polarity = int(data['polarity'])
        
        return jsonify({
            'success': True,
            'tac': detector.Tac,
            'tr': detector.Tr,
            'polarity': detector.specified_polarity
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Ellipse Detection Web Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print(f"Starting Ellipse Detection Server on http://{args.host}:{args.port}")
    print("Open your browser and navigate to the URL above")
    print("Press Ctrl+C to stop the server")
    
    app.run(host=args.host, port=args.port, debug=args.debug)
