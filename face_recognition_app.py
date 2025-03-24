import os
import cv2
import numpy as np
import gradio as gr
from deepface import DeepFace
import json
from datetime import datetime

# Create a directory to store authorized faces
AUTHORIZED_FACES_DIR = "authorized_faces"
if not os.path.exists(AUTHORIZED_FACES_DIR):
    os.makedirs(AUTHORIZED_FACES_DIR)

# File to store authorized face embeddings
EMBEDDINGS_FILE = "authorized_embeddings.json"

def load_authorized_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_authorized_embeddings(embeddings):
    with open(EMBEDDINGS_FILE, 'w') as f:
        json.dump(embeddings, f)

def get_face_embedding(image):
    try:
        embedding = DeepFace.represent(image, model_name="Facenet", enforce_detection=False)
        if embedding:
            return embedding[0]['embedding']
        return None
    except:
        return None

def register_face(image, name):
    if image is None:
        return "Please provide an image"
    
    embedding = get_face_embedding(image)
    if embedding is None:
        return "No face detected in the image"
    
    embeddings = load_authorized_embeddings()
    embeddings[name] = embedding
    save_authorized_embeddings(embeddings)
    
    # Save the image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(AUTHORIZED_FACES_DIR, f"{name}_{timestamp}.jpg")
    cv2.imwrite(image_path, image)
    
    return f"Successfully registered face for {name}"

def process_image(image, progress=gr.Progress()):
    if image is None:
        return None
    
    progress(0, desc="Loading authorized faces...")
    # Convert image to numpy array if it's not already
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Load authorized embeddings
    authorized_embeddings = load_authorized_embeddings()
    print(f"Loaded {len(authorized_embeddings)} authorized faces")
    
    try:
        progress(0.2, desc="Detecting faces...")
        # Use analyze instead of extract_faces for better face detection
        result = DeepFace.analyze(image, actions=['face'], enforce_detection=False)
        print(f"Analysis result: {result}")
        
        # Convert single result to list for consistency
        if isinstance(result, dict):
            result = [result]
            
        print(f"Number of faces detected: {len(result)}")
        
        # Process each face
        for i, face_data in enumerate(result):
            progress(0.2 + (i * 0.8 / len(result)), desc=f"Processing face {i+1}/{len(result)}")
            print(f"\nProcessing face {i+1}")
            
            # Get face region
            face_region = face_data.get('region', {})
            x = face_region.get('x', 0)
            y = face_region.get('y', 0)
            w = face_region.get('w', 0)
            h = face_region.get('h', 0)
            
            print(f"Face region: x={x}, y={y}, w={w}, h={h}")
            
            # Get face embedding
            face_embedding = get_face_embedding(image[y:y+h, x:x+w])
            if face_embedding is None:
                print(f"Could not get embedding for face {i+1}")
                continue
                
            # Convert embeddings to numpy arrays
            face_embedding = np.array(face_embedding)
            
            # Check if face is authorized
            is_authorized = False
            min_distance = float('inf')
            for name, auth_embedding in authorized_embeddings.items():
                auth_embedding = np.array(auth_embedding)
                distance = np.linalg.norm(face_embedding - auth_embedding)
                min_distance = min(min_distance, distance)
                if distance < 0.7:  # Threshold for face recognition
                    is_authorized = True
                    print(f"Face {i+1} matches {name} with distance {distance}")
                    break
            
            if not is_authorized:
                print(f"Face {i+1} is unauthorized (min distance: {min_distance})")
            
            # Ensure coordinates are within image bounds
            x = max(0, min(x, image.shape[1] - 1))
            y = max(0, min(y, image.shape[0] - 1))
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            # Draw rectangle with proper color
            color = (0, 255, 0) if is_authorized else (0, 0, 255)  # Green for authorized, Red for unauthorized
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            
    except Exception as e:
        print(f"Error during face detection: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    return image

def process_video(video):
    if video is None:
        return None
    
    # Process video frames
    cap = cv2.VideoCapture(video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame = process_image(frame)
        out.write(processed_frame)
    
    cap.release()
    out.release()
    return output_path

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Face Recognition System")
    
    with gr.Tab("Face Detection"):
        with gr.Row():
            input_image = gr.Image(label="Input Image")
            output_image = gr.Image(label="Processed Image")
        detect_btn = gr.Button("Detect Faces")
        detect_btn.click(fn=process_image, inputs=input_image, outputs=output_image)
    
    with gr.Tab("Video Detection"):
        with gr.Row():
            input_video = gr.Video(label="Input Video")
            output_video = gr.Video(label="Processed Video")
        process_video_btn = gr.Button("Process Video")
        process_video_btn.click(fn=process_video, inputs=input_video, outputs=output_video)
    
    with gr.Tab("Register New Face"):
        with gr.Row():
            register_image = gr.Image(label="Face Image")
            name_input = gr.Textbox(label="Name")
        register_btn = gr.Button("Register Face")
        register_output = gr.Textbox(label="Status")
        register_btn.click(fn=register_face, inputs=[register_image, name_input], outputs=register_output)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7868) 