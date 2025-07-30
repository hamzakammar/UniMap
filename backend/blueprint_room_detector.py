import cv2
import numpy as np
import networkx as nx
import pytesseract
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import re

# Configure pytesseract path (adjust for your system)
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'

IMAGE_PATH = 'Blueprints/1-1.png'
OUTPUT_IMAGE = 'Blueprints/1-1_labeled_rooms.png'
OUTPUT_GRAPH = 'Blueprints/1-1_labeled_rooms.gml'

def preprocess_for_ocr(image):
    """Preprocess image specifically for OCR text detection"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Use adaptive thresholding for better text contrast
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Invert so text is black on white background
    thresh = cv2.bitwise_not(thresh)
    
    return thresh

def extract_room_labels(image):
    """Extract room labels using OCR"""
    # Preprocess for OCR
    ocr_image = preprocess_for_ocr(image)
    
    # Configure OCR for better text detection
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    
    # Get OCR data with bounding boxes
    ocr_data = pytesseract.image_to_data(ocr_image, config=custom_config, output_type=pytesseract.Output.DICT)
    
    room_labels = []
    
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        conf = ocr_data['conf'][i]
        
        # Filter for likely room labels (numbers, alphanumeric)
        if text and conf > 30:  # Confidence threshold
            # Look for patterns like room numbers
            if re.match(r'^[0-9]{4}$', text) or re.match(r'^[0-9]{3}[A-Z]$', text) or re.match(r'^[0-9]{4}[A-Z]$', text):
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                # Calculate center
                center_x = x + w // 2
                center_y = y + h // 2
                
                room_labels.append({
                    'text': text,
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'confidence': conf
                })
    
    return room_labels

def detect_rooms_with_labels(image):
    """Detect rooms and associate them with OCR labels"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Preprocess for room detection - sensitive approach
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Use a more sensitive adaptive threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 7, 1)
    
    # Clean up the mask with smaller kernel for better detail
    kernel = np.ones((1,1), np.uint8)  # Minimal kernel to preserve details
    room_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_OPEN, kernel)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(room_mask, connectivity=8)
    
    rooms = []
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 300:  # Lower threshold to capture smaller rooms
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Calculate center
            cx = int(centroids[i][0])
            cy = int(centroids[i][1])
            
            # More permissive aspect ratio for different room shapes
            aspect_ratio = float(w) / h
            if 0.1 < aspect_ratio < 15.0:  # More permissive range
                # Minimal filtering - only exclude obvious non-rooms
                is_valid_room = True
                
                # Check for very small noise
                if area < 400 and (w < 8 or h < 8):
                    is_valid_room = False
                
                # Check for very thin corridors
                if (w < 6 or h < 6) and area < 800:
                    is_valid_room = False
                
                # Check for stair-like patterns (very restrictive)
                if aspect_ratio > 12.0 and area < 1000:
                    is_valid_room = False
                
                # Check for extremely large areas (likely multiple rooms)
                if area > 30000:
                    is_valid_room = False
                
                if is_valid_room:
                    rooms.append({
                        'center': (cx, cy),
                        'area': area,
                        'bbox': (x, y, w, h),
                        'label': i,
                        'dimensions': (w, h),
                        'aspect_ratio': aspect_ratio,
                        'room_label': None  # Will be filled by OCR matching
                    })
    
    return rooms

def match_labels_to_rooms(rooms, ocr_labels):
    """Match OCR labels to detected rooms based on proximity"""
    if not ocr_labels:
        return rooms
    
    # Create KDTree for efficient nearest neighbor search
    room_centers = np.array([room['center'] for room in rooms])
    label_centers = np.array([label['center'] for label in ocr_labels])
    
    if len(room_centers) == 0 or len(label_centers) == 0:
        return rooms
    
    tree = KDTree(label_centers)
    
    # Track which labels have been used
    used_labels = set()
    
    for i, room in enumerate(rooms):
        room_center = room['center']
        
        # Find the closest label within a reasonable distance
        distances, indices = tree.query([room_center], k=min(8, len(ocr_labels)))  # Check more labels
        distance = distances[0][0]  # Fix: get first element
        closest_label_idx = indices[0][0]  # Fix: get first element
        
        # If the closest label is within reasonable distance and not used, assign it
        if distance < 250 and closest_label_idx not in used_labels:  # Increased distance threshold
            room['room_label'] = ocr_labels[closest_label_idx]['text']
            room['label_confidence'] = ocr_labels[closest_label_idx]['confidence']
            room['label_center'] = ocr_labels[closest_label_idx]['center']
            used_labels.add(closest_label_idx)
        else:
            # Generate a fallback label
            room['room_label'] = f"Room_{i:03d}"
            room['label_confidence'] = 0
            room['label_center'] = room['center']
    
    return rooms

def detect_doors_and_connect_rooms(rooms, image):
    """Detect doors and create connections between rooms"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Preprocess for door detection
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 7, 2)
    
    # Find gaps (potential doors)
    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    gap_mask = cv2.absdiff(closed, opened)
    
    # Find door candidates with more restrictive criteria
    contours, _ = cv2.findContours(gap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    doors = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 1500:  # More restrictive area range
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            
            if 0.5 < aspect_ratio < 2.0:  # More restrictive aspect ratio
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    doors.append({
                        'center': (cx, cy),
                        'area': area,
                        'bbox': (x, y, w, h)
                    })
    
    # Create graph
    G = nx.Graph()
    
    # Add room nodes
    for i, room in enumerate(rooms):
        G.add_node(i, 
                  pos=str(room['center']),
                  area=int(room['area']),
                  label=str(room['room_label']),
                  dimensions=str(room['dimensions']),
                  aspect_ratio=float(room['aspect_ratio']),
                  bbox=str(room['bbox'])
                  )
    
    # Connect rooms through doors or proximity with more restrictive criteria
    if doors:
        # Cluster nearby doors with tighter clustering
        door_centers = np.array([door['center'] for door in doors])
        clustering = DBSCAN(eps=15, min_samples=1).fit(door_centers)  # Reduced eps from 25 to 15
        
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(doors[i])
        
        # Keep largest door from each cluster
        clustered_doors = []
        for cluster in clusters.values():
            if cluster:
                largest_door = max(cluster, key=lambda x: x['area'])
                clustered_doors.append(largest_door)
        
        # Connect rooms through doors with more restrictive distance
        for door in clustered_doors:
            door_center = door['center']
            connected_rooms = []
            
            for i, room in enumerate(rooms):
                room_center = room['center']
                distance = np.sqrt((door_center[0] - room_center[0])**2 + 
                                  (door_center[1] - room_center[1])**2)
                
                if distance < 80:  # Reduced from 120 to 80
                    connected_rooms.append(i)
            
            # Connect rooms through this door (limit to 2 rooms per door)
            for i in range(len(connected_rooms)):
                for j in range(i+1, min(i+2, len(connected_rooms))):  # Limit connections per door
                    room1, room2 = connected_rooms[i], connected_rooms[j]
                    if not G.has_edge(room1, room2):
                        weight = np.sqrt((rooms[room1]['center'][0] - rooms[room2]['center'][0])**2 + 
                                       (rooms[room1]['center'][1] - rooms[room2]['center'][1])**2)
                        G.add_edge(room1, room2, weight=float(weight), door_center=str(door_center), connection_type="door")
    
    # If no connections, use proximity with fewer connections
    if len(G.edges()) == 0:
        print("No door connections found, using proximity-based connections...")
        centers = [room['center'] for room in rooms]
        tree = KDTree(centers)
        
        for i, room in enumerate(rooms):
            center = room['center']
            dists, idxs = tree.query(center, k=min(3, len(rooms)))  # Reduced from 5 to 3
            
            for idx in idxs[1:]:
                if not G.has_edge(i, idx):
                    weight = np.sqrt((center[0] - centers[idx][0])**2 + 
                                   (center[1] - centers[idx][1])**2)
                    G.add_edge(i, idx, weight=float(weight), connection_type="proximity")
    
    return G, doors

def save_results(G, rooms, doors, filename):
    """Save graph and create summary"""
    # Save NetworkX graph
    nx.write_gml(G, filename)
    
    # Create summary file
    summary_filename = filename.replace('.gml', '_summary.txt')
    with open(summary_filename, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("BLUEPRINT ROOM DETECTION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("📊 DETECTION STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"• Total rooms detected: {len(rooms)}\n")
        f.write(f"• Total doors detected: {len(doors)}\n")
        f.write(f"• Total connections: {len(G.edges())}\n\n")
        
        f.write("🏷️ ROOM DETAILS:\n")
        f.write("-" * 30 + "\n")
        for i, room in enumerate(rooms):
            f.write(f"Room {i+1} ({room['room_label']}):\n")
            f.write(f"  📍 Center: {room['center']}\n")
            f.write(f"  📏 Area: {room['area']:,} pixels\n")
            f.write(f"  📐 Dimensions: {room['dimensions']}\n")
            f.write(f"  📊 Aspect ratio: {room['aspect_ratio']:.2f}\n")
            if 'label_confidence' in room:
                f.write(f"  🎯 Label confidence: {room['label_confidence']}%\n")
            f.write(f"  📦 Bounding box: {room['bbox']}\n\n")
        
        f.write("🔗 CONNECTIONS:\n")
        f.write("-" * 30 + "\n")
        for u, v, data in G.edges(data=True):
            room1_label = rooms[u]['room_label']
            room2_label = rooms[v]['room_label']
            connection_type = data.get('connection_type', 'unknown')
            weight = data.get('weight', 0)
            f.write(f"• {room1_label} ↔ {room2_label} ({connection_type}, weight={weight:.1f})\n")
        
        f.write(f"\n" + "=" * 60 + "\n")
        f.write("Generated by Blueprint Room Detector\n")
        f.write("=" * 60 + "\n")
    
    print(f"Graph saved as {filename}")
    print(f"Summary saved as {summary_filename}")

def visualize_results(image, rooms, doors, G):
    """Create visualization with room labels and connections"""
    result_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Draw rooms with labels
    for i, room in enumerate(rooms):
        center = room['center']
        room_label = room['room_label']
        
        # Color based on room size
        if room['area'] > 10000:
            color = (0, 0, 255)  # Red for large rooms
            radius = 12
            font_scale = 0.6
            thickness = 2
        elif room['area'] > 5000:
            color = (0, 255, 0)  # Green for medium rooms
            radius = 8
            font_scale = 0.5
            thickness = 1
        else:
            color = (255, 0, 0)  # Blue for small rooms
            radius = 6
            font_scale = 0.4
            thickness = 1
        
        cv2.circle(result_img, center, radius, color, -1)
        
        # Draw room label with better positioning and readability
        text_size = cv2.getTextSize(room_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = center[0] + radius + 5
        text_y = center[1] + text_size[1] // 2
        
        # Add white background for text readability
        cv2.rectangle(result_img, 
                     (text_x - 2, text_y - text_size[1] - 2),
                     (text_x + text_size[0] + 2, text_y + 2),
                     (255, 255, 255), -1)
        
        # Draw text with black color for contrast
        cv2.putText(result_img, room_label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        # Draw room bounding box
        x, y, w, h = room['bbox']
        cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 1)
    
    # Draw doors with better visibility
    for i, door in enumerate(doors):
        center = door['center']
        cv2.circle(result_img, center, 4, (0, 255, 255), -1)  # Yellow for doors
    
    # Draw connections with better visibility
    for u, v, data in G.edges(data=True):
        pos1 = G.nodes[u]['pos']
        pos2 = G.nodes[v]['pos']
        
        # Convert string back to tuple if needed
        if isinstance(pos1, str):
            pos1 = eval(pos1)
        if isinstance(pos2, str):
            pos2 = eval(pos2)
        
        cv2.line(result_img, pos1, pos2, (255, 0, 255), 2)
        
        # Draw door location if available
        if 'door_center' in data:
            door_center = data['door_center']
            if isinstance(door_center, str):
                door_center = eval(door_center)
            cv2.circle(result_img, door_center, 3, (0, 255, 255), -1)
    
    return result_img

def create_combined_visualization(image, rooms, doors, G):
    """Create a combined visualization showing blueprint and detected rooms"""
    # Create the room detection visualization
    detection_img = visualize_results(image, rooms, doors, G)
    
    # Create a side-by-side comparison
    # Get dimensions
    h, w = image.shape[:2]
    
    # Create a larger canvas for side-by-side display
    combined_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
    
    # Left side: Original blueprint
    blueprint_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    combined_img[:, :w] = blueprint_color
    
    # Right side: Room detection results
    combined_img[:, w:] = detection_img
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (255, 255, 255)
    
    # Label for original blueprint
    cv2.putText(combined_img, "Original Blueprint", (50, 50), 
               font, font_scale, color, thickness)
    
    # Label for room detection
    cv2.putText(combined_img, "Detected Rooms", (w + 50, 50), 
               font, font_scale, color, thickness)
    
    # Add legend
    legend_y = 100
    legend_spacing = 30
    
    # Large rooms (red)
    cv2.circle(combined_img, (30, legend_y), 8, (0, 0, 255), -1)
    cv2.putText(combined_img, "Large Rooms (>10k pixels)", (50, legend_y + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Medium rooms (green)
    cv2.circle(combined_img, (30, legend_y + legend_spacing), 6, (0, 255, 0), -1)
    cv2.putText(combined_img, "Medium Rooms (5k-10k pixels)", (50, legend_y + legend_spacing + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Small rooms (blue)
    cv2.circle(combined_img, (30, legend_y + 2*legend_spacing), 4, (255, 0, 0), -1)
    cv2.putText(combined_img, "Small Rooms (<5k pixels)", (50, legend_y + 2*legend_spacing + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Doors (yellow)
    cv2.circle(combined_img, (30, legend_y + 3*legend_spacing), 3, (0, 255, 255), -1)
    cv2.putText(combined_img, "Doors", (50, legend_y + 3*legend_spacing + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Connections (magenta)
    cv2.line(combined_img, (30, legend_y + 4*legend_spacing), (60, legend_y + 4*legend_spacing), 
             (255, 0, 255), 2)
    cv2.putText(combined_img, "Room Connections", (70, legend_y + 4*legend_spacing + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return combined_img

if __name__ == '__main__':
    # Load image
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    
    print("Processing blueprint for room detection and labeling...")
    
    # Extract room labels using OCR
    print("Extracting room labels with OCR...")
    ocr_labels = extract_room_labels(img)
    print(f"Found {len(ocr_labels)} potential room labels:")
    for label in ocr_labels:
        print(f"  {label['text']} at {label['center']} (confidence: {label['confidence']})")
    
    # Detect rooms
    print("\nDetecting rooms...")
    rooms = detect_rooms_with_labels(img)
    print(f"Detected {len(rooms)} rooms")
    
    # Match labels to rooms
    print("\nMatching labels to rooms...")
    rooms = match_labels_to_rooms(rooms, ocr_labels)
    
    # Detect doors and create connections
    print("\nDetecting doors and creating connections...")
    G, doors = detect_doors_and_connect_rooms(rooms, img)
    print(f"Detected {len(doors)} doors")
    
    # Save results
    save_results(G, rooms, doors, OUTPUT_GRAPH)
    
    # Create visualization
    result_img = visualize_results(img, rooms, doors, G)
    cv2.imwrite(OUTPUT_IMAGE, result_img)
    print(f"Visualization saved as {OUTPUT_IMAGE}")
    
    # Create combined visualization with blueprint
    combined_img = create_combined_visualization(img, rooms, doors, G)
    combined_output = OUTPUT_IMAGE.replace('.png', '_with_blueprint.png')
    cv2.imwrite(combined_output, combined_img)
    print(f"Combined visualization saved as {combined_output}")
    
    # Print summary
    print(f"\n=== FINAL RESULTS ===")
    print(f"Rooms detected: {len(rooms)}")
    print(f"Doors detected: {len(doors)}")
    print(f"Connections: {len(G.edges())}")
    
    print(f"\nRoom labels:")
    for i, room in enumerate(rooms):
        print(f"  {room['room_label']}: center={room['center']}, area={room['area']}")
    
    print(f"\nConnections:")
    for u, v, data in G.edges(data=True):
        room1_label = rooms[u]['room_label']
        room2_label = rooms[v]['room_label']
        conn_type = data.get('connection_type', 'unknown')
        print(f"  {room1_label} <-> {room2_label} ({conn_type})") 