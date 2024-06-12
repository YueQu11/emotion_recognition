import os
import cv2
import numpy as np
from keras.models import load_model
from flask import Flask, Response, render_template_string

# 加载本地模型文件
model = load_model('emotion_model.h5')  # 请确保路径正确

# 定义情绪标签
emotion_labels = ['surprise', 'sad', 'neutral', 'happy', 'fear', 'disgust', 'angry']

# 初始化Flask应用
app = Flask(__name__)

# HTML模板字符串
html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion Recognition</title>
</head>
<body>
    <h1>Emotion Recognition Video Stream</h1>
    <img src="{{ url_for('video_feed') }}" width="640" height="480">
</body>
</html>
'''


# 根URL的路由
@app.route('/')
def index():
    return render_template_string(html_template)


# 摄像头捕获函数
def gen_frames():
    try:
        # 确认文件路径是否正确
        cascade_path = 'models/haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            print(f"File not found: {cascade_path}")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        face_cascade = cv2.CascadeClassifier(cascade_path)

        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to capture image")
                break
            else:
                print("Image captured")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                if faces is None:
                    print("No faces detected")
                else:
                    print(f"Faces detected: {len(faces)}")

                for (x, y, w, h) in faces:
                    print(f"Face coordinates: x={x}, y={y}, w={w}, h={h}")
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    roi_gray = roi_gray.astype('float32') / 255.0
                    roi_gray = np.expand_dims(roi_gray, axis=0)
                    roi_gray = np.expand_dims(roi_gray, axis=-1)

                    prediction = model.predict(roi_gray)
                    max_index = np.argmax(prediction[0])
                    emotion = emotion_labels[max_index]

                    # 确保坐标在图像范围内
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(frame.shape[1], x + w)
                    y2 = min(frame.shape[0], y + h)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    print("Failed to encode image")
                    continue
                frame = buffer.tobytes()
                print("Frame encoded and sent")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"Error occurred: {e}")


# 定义视频流的路由
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# 启动Flask应用
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
