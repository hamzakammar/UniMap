import cv2
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

IMAGE_PATH = 'Blueprints/1-1.png'
OUTPUT_IMAGE = 'Blueprints/1-1_advanced_doors.png'
OUTPUT_GRAPH = 'Blueprints/1-1_advanced_doors.gml'

def preprocess_image(image_path):
    """Preprocess image for advanced door detection"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Apply Gaussian blur with smaller kernel
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Use adaptive thresholding with smaller block size
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 7, 2)
    
    return img, thresh

def detect_walls_and_gaps(thresh_img):
    """Detect walls and find gaps using multiple techniques"""
    # Method 1: Hough Transform for lines
    edges = cv2.Canny(thresh_img, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=1000, 
                           minLineLength=None, maxLineGap=0)
    
    # Method 2: Morphological operations to find gaps
    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
    
    # The difference between closed and opened should show gaps
    gap_mask = cv2.absdiff(closed, opened)
    
    return edges, lines, gap_mask

def find_door_candidates_advanced(thresh_img, gap_mask, wall_lines):
    """Advanced door detection using multiple techniques"""
    door_candidates = []
    
    # Technique 1: Find gaps in morphological operations
    contours, _ = cv2.findContours(gap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 10 < area < 2000:  # Door-sized areas
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            
            if 0.3 < aspect_ratio < 2.5:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    door_candidates.append({
                        'center': (cx, cy),
                        'area': area,
                        'bbox': (x, y, w, h),
                        'contour': cnt,
                        'method': 'morphological'
                    })
    
    # Technique 2: Look for small white regions near wall lines
    if wall_lines is not None:
        wall_mask = np.zeros_like(thresh_img)
        for line in wall_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(wall_mask, (x1, y1), (x2, y2), 255, 3)
        
        # Dilate wall mask to find regions near walls
        kernel = np.ones((5,5), np.uint8)
        dilated_walls = cv2.dilate(wall_mask, kernel, iterations=1)
        
        # Find white regions near walls
        near_walls = cv2.bitwise_and(thresh_img, dilated_walls)
        
        contours, _ = cv2.findContours(near_walls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 1500:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                
                if 0.2 < aspect_ratio < 3.0:
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        
                        # Check if this is not too close to existing candidates
                        too_close = False
                        for existing in door_candidates:
                            dist = np.sqrt((cx - existing['center'][0])**2 + (cy - existing['center'][1])**2)
                            if dist < 30:
                                too_close = True
                                break
                        
                        if not too_close:
                            door_candidates.append({
                                'center': (cx, cy),
                                'area': area,
                                'bbox': (x, y, w, h),
                                'contour': cnt,
                                'method': 'near_walls'
                            })
    
    return door_candidates

def detect_rooms_improved(thresh_img):
    """Improved room detection using connected components"""
    # Invert so walls are black
    room_mask = cv2.bitwise_not(thresh_img)
    
    # Remove small noise with smaller kernel
    kernel = np.ones((3,3), np.uint8)
    room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_OPEN, kernel)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(room_mask, connectivity=8)
    
    rooms = []
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 500:  # Much lower minimum area threshold
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Calculate center
            cx = int(centroids[i][0])
            cy = int(centroids[i][1])
            
            # More permissive aspect ratio filter
            aspect_ratio = float(w) / h
            if 0.1 < aspect_ratio < 10.0:  # Much wider range
                # Determine room type based on size and aspect ratio
                if area > 10000:
                    room_type = "large_room"
                elif area > 5000:
                    room_type = "medium_room"
                else:
                    room_type = "small_room"
                
                # Generate room label
                room_label = f"Room_{len(rooms):03d}"
                
                rooms.append({
                    'center': (cx, cy),
                    'area': area,
                    'bbox': (x, y, w, h),
                    'label': i,
                    'room_type': room_type,
                    'room_label': room_label,
                    'dimensions': (w, h),
                    'aspect_ratio': aspect_ratio
                })
    
    return rooms

def connect_rooms_with_doors(rooms, doors, max_door_distance=80):
    """Connect rooms using detected doors"""
    G = nx.Graph()
    
    # Add room nodes with detailed attributes
    for i, room in enumerate(rooms):
        G.add_node(i, 
                  pos=str(room['center']),  # Convert tuple to string
                  area=int(room['area']),
                  label=str(room['room_label']),
                  room_type=str(room['room_type']),
                  dimensions=str(room['dimensions']),  # Convert tuple to string
                  aspect_ratio=float(room['aspect_ratio']),
                  bbox=str(room['bbox'])  # Convert tuple to string
                  )
    
    if not doors:
        print("No doors detected, using proximity-based connections...")
        # Connect by proximity
        centers = [room['center'] for room in rooms]
        tree = KDTree(centers)
        
        for i, room in enumerate(rooms):
            center = room['center']
            dists, idxs = tree.query(center, k=min(5, len(rooms)))  # Increased from 3 to 5 neighbors
            
            for idx in idxs[1:]:
                if not G.has_edge(i, idx):
                    weight = np.sqrt((center[0] - centers[idx][0])**2 + 
                                   (center[1] - centers[idx][1])**2)
                    G.add_edge(i, idx, weight=float(weight), connection_type="proximity")
        return G
    
    # Connect through doors
    for door in doors:
        door_center = door['center']
        
        # Find rooms near this door
        connected_rooms = []
        for i, room in enumerate(rooms):
            room_center = room['center']
            distance = np.sqrt((door_center[0] - room_center[0])**2 + 
                              (door_center[1] - room_center[1])**2)
            
            if distance < max_door_distance:
                connected_rooms.append(i)
        
        # Connect rooms through this door
        for i in range(len(connected_rooms)):
            for j in range(i+1, len(connected_rooms)):
                room1, room2 = connected_rooms[i], connected_rooms[j]
                if not G.has_edge(room1, room2):
                    weight = np.sqrt((rooms[room1]['center'][0] - rooms[room2]['center'][0])**2 + 
                                   (rooms[room1]['center'][1] - rooms[room2]['center'][1])**2)
                    G.add_edge(room1, room2, weight=float(weight), door_center=str(door_center), connection_type="door")
    
    # If no connections were made through doors, use proximity as fallback
    if len(G.edges()) == 0 and len(doors) > 0:
        print("No door connections made, using proximity fallback...")
        centers = [room['center'] for room in rooms]
        tree = KDTree(centers)
        
        for i, room in enumerate(rooms):
            center = room['center']
            dists, idxs = tree.query(center, k=min(5, len(rooms)))
            
            for idx in idxs[1:]:
                if not G.has_edge(i, idx):
                    weight = np.sqrt((center[0] - centers[idx][0])**2 + 
                                   (center[1] - centers[idx][1])**2)
                    G.add_edge(i, idx, weight=float(weight), connection_type="proximity_fallback")
    
    return G

def save_graph_with_labels(G, rooms, filename):
    """Save graph with detailed room information"""
    # Save the NetworkX graph
    nx.write_gml(G, filename)
    
    # Create a summary file
    summary_filename = filename.replace('.gml', '_summary.txt')
    with open(summary_filename, 'w') as f:
        f.write("ROOM DETECTION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total rooms detected: {len(rooms)}\n")
        f.write(f"Total connections: {len(G.edges())}\n\n")
        
        f.write("ROOM DETAILS:\n")
        f.write("-" * 30 + "\n")
        for i, room in enumerate(rooms):
            f.write(f"Room {i} ({room['room_label']}):\n")
            f.write(f"  Center: {room['center']}\n")
            f.write(f"  Area: {room['area']} pixels\n")
            f.write(f"  Dimensions: {room['dimensions']}\n")
            f.write(f"  Type: {room['room_type']}\n")
            f.write(f"  Aspect ratio: {room['aspect_ratio']:.2f}\n")
            f.write(f"  Bounding box: {room['bbox']}\n\n")
        
        f.write("CONNECTIONS:\n")
        f.write("-" * 30 + "\n")
        for u, v, data in G.edges(data=True):
            room1_label = rooms[u]['room_label']
            room2_label = rooms[v]['room_label']
            connection_type = data.get('connection_type', 'unknown')
            weight = data.get('weight', 0)
            f.write(f"{room1_label} <-> {room2_label} ({connection_type}, weight={weight:.1f})\n")
    
    print(f"Graph saved as {filename}")
    print(f"Summary saved as {summary_filename}")

def print_graph_info(G, rooms):
    """Print detailed information about the graph"""
    print(f"\n=== GRAPH INFORMATION ===")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    
    # Count room types
    room_types = {}
    for room in rooms:
        room_type = room['room_type']
        room_types[room_type] = room_types.get(room_type, 0) + 1
    
    print(f"\nRoom types:")
    for room_type, count in room_types.items():
        print(f"  {room_type}: {count}")
    
    # Count connection types
    connection_types = {}
    for u, v, data in G.edges(data=True):
        conn_type = data.get('connection_type', 'unknown')
        connection_types[conn_type] = connection_types.get(conn_type, 0) + 1
    
    print(f"\nConnection types:")
    for conn_type, count in connection_types.items():
        print(f"  {conn_type}: {count}")
    
    # Find isolated rooms
    isolated_rooms = list(nx.isolates(G))
    if isolated_rooms:
        print(f"\nIsolated rooms: {len(isolated_rooms)}")
        for node in isolated_rooms:
            room_label = rooms[node]['room_label']
            print(f"  {room_label}")
    else:
        print(f"\nNo isolated rooms - all rooms are connected!")
    
    # Find connected components
    components = list(nx.connected_components(G))
    if len(components) > 1:
        print(f"\nConnected components: {len(components)}")
        for i, component in enumerate(components):
            room_labels = [rooms[node]['room_label'] for node in component]
            print(f"  Component {i+1}: {', '.join(room_labels)}")
    else:
        print(f"\nAll rooms are in one connected component!")

if __name__ == '__main__':
    # Load and preprocess
    img, thresh = preprocess_image(IMAGE_PATH)
    edges, wall_lines, gap_mask = detect_walls_and_gaps(thresh)
    
    # Detect doors using advanced methods
    door_candidates = find_door_candidates_advanced(thresh, gap_mask, wall_lines)
    print(f"Found {len(door_candidates)} door candidates")
    
    # Cluster nearby doors
    if door_candidates:
        centers = np.array([door['center'] for door in door_candidates])
        clustering = DBSCAN(eps=25, min_samples=1).fit(centers)  # Reduced eps from 40 to 25
        
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(door_candidates[i])
        
        doors = []
        for cluster in clusters.values():
            if cluster:
                largest_door = max(cluster, key=lambda x: x['area'])
                doors.append(largest_door)
        
        print(f"After clustering: {len(doors)} doors")
    else:
        doors = []
        print("No doors detected")
    
    # Detect rooms
    rooms = detect_rooms_improved(thresh)
    print(f"Detected {len(rooms)} rooms")
    
    # Build graph
    G = connect_rooms_with_doors(rooms, doors, max_door_distance=120)  # Much more aggressive door distance
    save_graph_with_labels(G, rooms, OUTPUT_GRAPH)
    print_graph_info(G, rooms)
    
    # Visualize
    result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw rooms with labels
    for i, room in enumerate(rooms):
        center = room['center']
        room_label = room['room_label']
        
        # Color based on room type
        if room['room_type'] == 'large_room':
            color = (0, 0, 255)  # Red
            radius = 12
        elif room['room_type'] == 'medium_room':
            color = (0, 255, 0)  # Green
            radius = 8
        else:
            color = (255, 0, 0)  # Blue
            radius = 6
        
        cv2.circle(result_img, center, radius, color, -1)
        cv2.putText(result_img, room_label, (center[0]+radius+2, center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw room bounding box
        x, y, w, h = room['bbox']
        cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 1)
    
    # Draw doors with different colors for different methods
    for i, door in enumerate(doors):
        center = door['center']
        method = door.get('method', 'unknown')
        
        if method == 'morphological':
            color = (255, 255, 0)  # Yellow
        elif method == 'near_walls':
            color = (0, 255, 255)  # Cyan
        else:
            color = (255, 0, 255)  # Magenta
        
        cv2.circle(result_img, center, 6, color, -1)
        cv2.putText(result_img, f"D{i}", (center[0]+8, center[1]-8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw connections
    for u, v, data in G.edges(data=True):
        # Get positions - handle both string and tuple formats
        pos1 = G.nodes[u]['pos']
        pos2 = G.nodes[v]['pos']
        
        # Convert string back to tuple if needed
        if isinstance(pos1, str):
            pos1 = eval(pos1)  # Convert "(x, y)" string back to tuple
        if isinstance(pos2, str):
            pos2 = eval(pos2)  # Convert "(x, y)" string back to tuple
        
        cv2.line(result_img, pos1, pos2, (255, 0, 255), 2)
        
        # Draw door location if available
        if 'door_center' in data:
            door_center = data['door_center']
            if isinstance(door_center, str):
                door_center = eval(door_center)  # Convert string back to tuple
            cv2.circle(result_img, door_center, 4, (0, 255, 255), -1)
    
    cv2.imwrite(OUTPUT_IMAGE, result_img)
    print(f"Visualization saved as {OUTPUT_IMAGE}")
    
    # Save debug images
    cv2.imwrite('Blueprints/1-1_gap_mask_advanced.png', gap_mask)
    cv2.imwrite('Blueprints/1-1_edges_advanced.png', edges)
    print("Debug images saved")
    
    # Print detailed results
    print(f"\nDetected {len(rooms)} rooms:")
    for i, room in enumerate(rooms):
        print(f"  {room['room_label']}: center={room['center']}, area={room['area']}, type={room['room_type']}")
    
    print(f"\nDetected {len(doors)} doors:")
    for i, door in enumerate(doors):
        print(f"  Door {i}: center={door['center']}, method={door.get('method', 'unknown')}")
    
    print(f"\nRoom connections:")
    for u, v, data in G.edges(data=True):
        room1_label = rooms[u]['room_label']
        room2_label = rooms[v]['room_label']
        conn_type = data.get('connection_type', 'unknown')
        print(f"  {room1_label} <-> {room2_label} ({conn_type})") 