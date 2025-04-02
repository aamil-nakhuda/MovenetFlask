from flask import Flask, Response, render_template_string
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# 1. Load the TFLite Model
interpreter = tf.lite.Interpreter(model_path='3.tflite')
interpreter.allocate_tensors()

# 2. Open the Video Capture
cap = cv2.VideoCapture(0)

# 3. Define Drawing Functions
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1 > confidence_threshold) and (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

# 4. Create a Generator Function for Video Streaming
def generate_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for the model:
        img = frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.float32)

        # Setup input and output details for the interpreter:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Run inference:
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        # Draw keypoints and connections:
        draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
        draw_keypoints(frame, keypoints_with_scores, 0.4)

        # Encode the frame in JPEG format:
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        # Yield the output frame in the multipart response format:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# 5. Set Up Flask Routes
# Main page that renders the video stream
@app.route('/')
def index():
    # Using render_template_string for simplicity.
    # You can also use a separate HTML template file.
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>MoveNet Pose Estimation</title>
        </head>
        <body>
            <h1>Real-Time Pose Estimation</h1>
            <img src="{{ url_for('video_feed') }}" width="640" height="480">
        </body>
        </html>
    ''')

# Video streaming route.
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 6. Clean Up on Exit
@app.route('/shutdown')
def shutdown():
    cap.release()
    cv2.destroyAllWindows()
    return "Server shutting down..."

# 7. Run the Flask Application
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

