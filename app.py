from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import os
import sys

# Add the backend directory to the path to import blueprint_room_detector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from blueprint_room_detector import detect_rooms_with_labels, match_labels_to_rooms, extract_room_labels, detect_doors_and_connect_rooms
    BLUEPRINT_DETECTION_AVAILABLE = True
except ImportError:
    BLUEPRINT_DETECTION_AVAILABLE = False
    print("Warning: Blueprint room detection not available, using hardcoded rooms")

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables to store detected rooms and graph
detected_rooms = []
detected_graph = None

def load_blueprint_rooms():
    """Load rooms from blueprint detection if available"""
    global detected_rooms, detected_graph
    
    if not BLUEPRINT_DETECTION_AVAILABLE:
        return False
    
    try:
        import cv2
        
        # Load the blueprint image
        blueprint_path = 'Blueprints/1-1.png'
        if not os.path.exists(blueprint_path):
            print(f"Blueprint not found: {blueprint_path}")
            return False
        
        img = cv2.imread(blueprint_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Failed to load blueprint image")
            return False
        
        # Extract room labels using OCR
        print("Extracting room labels from blueprint...")
        ocr_labels = extract_room_labels(img)
        print(f"Found {len(ocr_labels)} room labels")
        
        # Detect rooms
        print("Detecting rooms from blueprint...")
        rooms = detect_rooms_with_labels(img)
        print(f"Detected {len(rooms)} rooms")
        
        # Match labels to rooms
        print("Matching labels to rooms...")
        rooms = match_labels_to_rooms(rooms, ocr_labels)
        
        # Detect doors and create connections
        print("Detecting doors and creating connections...")
        G, doors = detect_doors_and_connect_rooms(rooms, img)
        print(f"Created graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
        
        detected_rooms = rooms
        detected_graph = G
        
        print("Successfully loaded blueprint rooms!")
        return True
        
    except Exception as e:
        print(f"Error loading blueprint rooms: {e}")
        return False

# Try to load blueprint rooms on startup
blueprint_loaded = load_blueprint_rooms()

# Fallback to hardcoded rooms if blueprint detection fails
if not blueprint_loaded:
    print("Using hardcoded room data...")
    
    # Hardcoded room data (fallback)
    detected_rooms = [
        {'room_label': '1001', 'center': (100, 100)},
        {'room_label': '1002', 'center': (200, 100)},
        {'room_label': '1003', 'center': (300, 100)},
        {'room_label': '1004', 'center': (400, 100)},
        {'room_label': '1006', 'center': (100, 200)},
        {'room_label': '1007', 'center': (200, 200)},
        {'room_label': '1008', 'center': (300, 200)},
        {'room_label': '1009', 'center': (400, 200)},
        {'room_label': '1011', 'center': (100, 300)},
        {'room_label': '1012', 'center': (200, 300)},
        {'room_label': '1802', 'center': (300, 300)},
        {'room_label': '1902', 'center': (400, 300)},
    ]
    
    # Create a simple graph for hardcoded rooms
    detected_graph = nx.Graph()
    for i, room in enumerate(detected_rooms):
        detected_graph.add_node(i, pos=room['center'], label=room['room_label'])
    
    # Add some connections
    for i in range(len(detected_rooms) - 1):
        detected_graph.add_edge(i, i + 1, weight=100.0)

@app.route('/')
def home():
    return jsonify({
        "message": "UniMap API",
        "status": "running",
        "blueprint_loaded": blueprint_loaded,
        "rooms_available": len(detected_rooms),
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/<start>/<end>": "Get route between rooms"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "blueprint_loaded": blueprint_loaded})

@app.route('/<start>/<end>')
def get_route(start, end):
    try:
        # Find room indices
        start_idx = None
        end_idx = None
        
        for i, room in enumerate(detected_rooms):
            if room['room_label'] == start:
                start_idx = i
            if room['room_label'] == end:
                end_idx = i
        
        if start_idx is None:
            return jsonify({"error": f"Start room '{start}' not found"}), 404
        if end_idx is None:
            return jsonify({"error": f"End room '{end}' not found"}), 404
        
        # Find shortest path
        try:
            path = nx.shortest_path(detected_graph, start_idx, end_idx, weight='weight')
            path_rooms = [detected_rooms[i]['room_label'] for i in path]
        except nx.NetworkXNoPath:
            return jsonify({"error": f"No path found between '{start}' and '{end}'"}), 404
        
        # Create route visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw nodes
        for i, room in enumerate(detected_rooms):
            pos = room['center']
            ax.plot(pos[0], pos[1], 'o', markersize=10, color='blue')
            ax.text(pos[0] + 10, pos[1], room['room_label'], fontsize=8)
        
        # Draw edges
        for u, v, data in detected_graph.edges(data=True):
            pos1 = detected_rooms[u]['center']
            pos2 = detected_rooms[v]['center']
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'gray', alpha=0.5)
        
        # Highlight path
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            pos1 = detected_rooms[u]['center']
            pos2 = detected_rooms[v]['center']
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'red', linewidth=3)
        
        # Highlight start and end nodes
        start_pos = detected_rooms[start_idx]['center']
        end_pos = detected_rooms[end_idx]['center']
        ax.plot(start_pos[0], start_pos[1], 'o', markersize=15, color='green', label='Start')
        ax.plot(end_pos[0], end_pos[1], 'o', markersize=15, color='red', label='End')
        
        ax.set_title(f'Route from {start} to {end}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save the plot
        img_path = './static/abc.png'
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return send_from_directory('./static', 'abc.png')
        
    except Exception as e:
        return jsonify({"error": f"Error generating route: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting UniMap Flask server...")
    print(f"Blueprint detection: {'Available' if BLUEPRINT_DETECTION_AVAILABLE else 'Not available'}")
    print(f"Blueprint loaded: {blueprint_loaded}")
    print(f"Rooms available: {len(detected_rooms)}")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 