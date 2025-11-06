# from flask import Flask, request, render_template, jsonify
# import numpy as np
# import mediapipe as mp
# import cv2
# from tensorflow.keras.models import load_model
# import joblib
# import os

# # ------------------ Flask App ------------------
# app = Flask(__name__)

# # ------------------ Load Model and Scaler ------------------
# model = load_model("pose_classification_model.h5")
# scaler = joblib.load("pose_scaler.pkl")

# # ------------------ Mediapipe Setup ------------------
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# # ------------------ Pose Labels ------------------
# pose_labels = {
#     0: 'Virabhadrasana Three', 1: 'Phalakasana', 2: 'Bitilasana',
#     3: 'Uttanasana', 4: 'Vrksasana', 5: 'Adho Mukha Svanasana',
#     6: 'Padmasana', 7: 'Virabhadrasana One', 8: 'Garudasana',
#     9: 'Salamba Sarvangasana', 10: 'Bakasana', 11: 'Utkatasana',
#     12: 'Utthita Parsvakonasana', 13: 'Anjaneyasana', 14: 'Ardha Matsyendrasana',
#     15: 'Halasana', 16: 'Baddha Konasana', 17: 'Utthita Hasta Padangusthasana',
#     18: 'Balasana', 19: 'Vasisthasana', 20: 'Urdhva Mukha Svsnssana',
#     21: 'Ustrasana', 22: 'Camatkarasana', 23: 'Setu Bandha Sarvangasana',
#     24: 'Malasana'
# }

# # ------------------ Utility Functions ------------------
# def calculate_angle(a, b, c):
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     ba = a - b
#     bc = c - b
#     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#     return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# def calculate_distance(a, b):
#     return np.linalg.norm(np.array(a) - np.array(b))

# def extract_pose_features(landmarks):
#     def L(p): return [landmarks[p].x, landmarks[p].y, landmarks[p].z]

#     # Keypoints
#     A11, A12 = L(mp_pose.PoseLandmark.LEFT_SHOULDER), L(mp_pose.PoseLandmark.RIGHT_SHOULDER)
#     A13, A14 = L(mp_pose.PoseLandmark.LEFT_ELBOW), L(mp_pose.PoseLandmark.RIGHT_ELBOW)
#     A15, A16 = L(mp_pose.PoseLandmark.LEFT_WRIST), L(mp_pose.PoseLandmark.RIGHT_WRIST)
#     A17, A18 = L(mp_pose.PoseLandmark.LEFT_PINKY), L(mp_pose.PoseLandmark.RIGHT_PINKY)
#     A21, A22 = L(mp_pose.PoseLandmark.LEFT_THUMB), L(mp_pose.PoseLandmark.RIGHT_THUMB)
#     A23, A24 = L(mp_pose.PoseLandmark.LEFT_HIP), L(mp_pose.PoseLandmark.RIGHT_HIP)
#     A25, A26 = L(mp_pose.PoseLandmark.LEFT_KNEE), L(mp_pose.PoseLandmark.RIGHT_KNEE)
#     A27, A28 = L(mp_pose.PoseLandmark.LEFT_ANKLE), L(mp_pose.PoseLandmark.RIGHT_ANKLE)
#     A31, A32 = L(mp_pose.PoseLandmark.LEFT_FOOT_INDEX), L(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)

#     # Angles
#     angles = [
#         calculate_angle(A14, A12, A24), calculate_angle(A14, A12, A11),
#         calculate_angle(A13, A11, A23), calculate_angle(A12, A11, A13),
#         calculate_angle(A16, A14, A12), calculate_angle(A15, A13, A11),
#         calculate_angle(A12, A24, A26), calculate_angle(A11, A23, A25),
#         calculate_angle(A28, A26, A24), calculate_angle(A23, A25, A27),
#         calculate_angle(A32, A28, A26), calculate_angle(A31, A27, A25),
#         calculate_angle(A22, A16, A18), calculate_angle(A21, A15, A17)
#     ]

#     # Distances
#     dist_data = [
#         calculate_distance(A16, A15), calculate_distance(A27, A28),
#         calculate_distance(A16, A28), calculate_distance(A27, A15),
#         calculate_distance(A16, A27), calculate_distance(A28, A15),
#         calculate_distance(A12, A26), calculate_distance(A11, A25),
#         calculate_distance(A11, A16), calculate_distance(A12, A15)
#     ]
#     return np.array(angles + dist_data, dtype=np.float32)

# # ------------------ Routes ------------------
# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict_pose():
#     if "image" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["image"]
#     img_path = "uploaded.jpg"
#     file.save(img_path)

#     image = cv2.imread(img_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = pose.process(image_rgb)

#     if not results.pose_landmarks:
#         return jsonify({"pose": "No pose detected", "confidence": 0.0})

#     landmarks = results.pose_landmarks.landmark
#     X = extract_pose_features(landmarks)
#     X_scaled = scaler.transform([X])

#     prediction = model.predict(X_scaled)
#     pose_index = np.argmax(prediction)
#     confidence = float(np.max(prediction))
#     predicted_label = pose_labels.get(pose_index, "Unknown Pose")

#     CONFIDENCE_THRESHOLD = 0.75
#     if confidence < CONFIDENCE_THRESHOLD:
#         predicted_label = "Uncertain / No pose detected"

#     return jsonify({
#         "pose": predicted_label,
#         "confidence": round(confidence, 3)
#     })

# # ------------------ Run App ------------------
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
from flask import Flask, render_template, request, url_for
import numpy as np
import mediapipe as mp
import cv2
from tensorflow.keras.models import load_model
import joblib
import os
import uuid
import traceback

# ------------------- Flask Setup -------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ------------------- Load Model & Scaler -------------------
model = load_model("pose_classification_model.h5")
scaler = joblib.load("pose_scaler.pkl")

# ------------------- Mediapipe Setup -------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# ------------------- Pose Labels -------------------
pose_labels = {
    0: 'Virabhadrasana Three', 1: 'Phalakasana', 2: 'Bitilasana',
    3: 'Uttanasana', 4: 'Vrksasana', 5: 'Adho Mukha Svanasana',
    6: 'Padmasana', 7: 'Virabhadrasana One', 8: 'Garudasana',
    9: 'Salamba Sarvangasana', 10: 'Bakasana', 11: 'Utkatasana',
    12: 'Utthita Parsvakonasana', 13: 'Anjaneyasana', 14: 'Ardha Matsyendrasana',
    15: 'Halasana', 16: 'Baddha Konasana', 17: 'Utthita Hasta Padangusthasana',
    18: 'Balasana', 19: 'Vasisthasana', 20: 'Urdhva Mukha Svsnssana',
    21: 'Ustrasana', 22: 'Camatkarasana', 23: 'Setu Bandha Sarvangasana',
    24: 'Malasana'
}

# ------------------- Utility Functions -------------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def calculate_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def extract_pose_features(landmarks):
    def L(p):
        return [landmarks[p].x, landmarks[p].y, landmarks[p].z]

    # Keypoints
    A11, A12 = L(mp_pose.PoseLandmark.LEFT_SHOULDER), L(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    A13, A14 = L(mp_pose.PoseLandmark.LEFT_ELBOW), L(mp_pose.PoseLandmark.RIGHT_ELBOW)
    A15, A16 = L(mp_pose.PoseLandmark.LEFT_WRIST), L(mp_pose.PoseLandmark.RIGHT_WRIST)
    A17, A18 = L(mp_pose.PoseLandmark.LEFT_PINKY), L(mp_pose.PoseLandmark.RIGHT_PINKY)
    A21, A22 = L(mp_pose.PoseLandmark.LEFT_THUMB), L(mp_pose.PoseLandmark.RIGHT_THUMB)
    A23, A24 = L(mp_pose.PoseLandmark.LEFT_HIP), L(mp_pose.PoseLandmark.RIGHT_HIP)
    A25, A26 = L(mp_pose.PoseLandmark.LEFT_KNEE), L(mp_pose.PoseLandmark.RIGHT_KNEE)
    A27, A28 = L(mp_pose.PoseLandmark.LEFT_ANKLE), L(mp_pose.PoseLandmark.RIGHT_ANKLE)
    A31, A32 = L(mp_pose.PoseLandmark.LEFT_FOOT_INDEX), L(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)

    # ---- Angles ----
    angles = [
        calculate_angle(A14, A12, A24),
        calculate_angle(A14, A12, A11),
        calculate_angle(A13, A11, A23),
        calculate_angle(A12, A11, A13),
        calculate_angle(A16, A14, A12),
        calculate_angle(A15, A13, A11),
        calculate_angle(A12, A24, A26),
        calculate_angle(A11, A23, A25),
        calculate_angle(A28, A26, A24),
        calculate_angle(A23, A25, A27),
        calculate_angle(A32, A28, A26),
        calculate_angle(A31, A27, A25),
        calculate_angle(A22, A16, A18),
        calculate_angle(A21, A15, A17)
    ]

    # ---- Distances ----
    dist_data = [
        calculate_distance(A16, A15),
        calculate_distance(A27, A28),
        calculate_distance(A16, A28),
        calculate_distance(A27, A15),
        calculate_distance(A16, A27),
        calculate_distance(A28, A15),
        calculate_distance(A12, A26),
        calculate_distance(A11, A25),
        calculate_distance(A11, A16),
        calculate_distance(A12, A15)
    ]
    return np.array(angles + dist_data, dtype=np.float32)

# ------------------- Routes -------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('index.html', error="No file selected.")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected.")

        # Save uploaded file
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Read and process image
        image = cv2.imread(filepath)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            return render_template('index.html',
                                   uploaded_image=filepath,
                                   error="No pose detected in the image.")

        # Draw landmarks
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        # Extract features safely
        landmarks = results.pose_landmarks.landmark
        if len(landmarks) < 33:
            return render_template('index.html', uploaded_image=filepath,
                                   error="Incomplete landmarks detected.")

        X = extract_pose_features(landmarks)
        if X.shape[0] != 24:
            return render_template('index.html', uploaded_image=filepath,
                                   error=f"Invalid feature shape: {X.shape[0]} instead of 24.")

        # Scale and predict
        X_scaled = scaler.transform([X])
        prediction = model.predict(X_scaled)
        pose_index = np.argmax(prediction)
        confidence = np.max(prediction)
        predicted_label = pose_labels.get(pose_index, "Unknown Pose")


        #  ------------------ Confidence Threshold ------------------
        CONFIDENCE_THRESHOLD = 0.75
        if confidence < CONFIDENCE_THRESHOLD:
            predicted_label = "Uncertain / No pose detected"
        else:
            predicted_label = pose_labels.get(pose_index, "Unknown Pose")
        # ----------------------------------------------------------

        # Save output image with landmarks
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        # cv2.putText(image, f"{predicted_label} ({confidence*100:.2f}%)", (30, 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.imwrite(output_path, image)


        return render_template('index.html',
                       uploaded_image=filename,   # just filename
                       output_image=filename,     # just filename
                       predicted_label=predicted_label,
                       confidence=f"{confidence*100:.2f}%")

    except Exception as e:
        traceback.print_exc()
        return render_template('index.html', error=f"Error: {e}")

# ------------------- Run App -------------------
if __name__ == '__main__':
    app.run(debug=True)
