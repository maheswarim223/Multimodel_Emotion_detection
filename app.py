import os
import sys
import json
import time
import queue
import threading
import subprocess
import io
import traceback
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── per-job queues ────────────────────────────────────────────────────────────
job_queues: dict[str, queue.Queue] = {}
job_lock = threading.Lock()

# ── model cache ──────────────────────────────────────────────────────────────
models_cache = {
    'text': None,
    'text_vec': None,
    'voice': None,
    'voice_le': None,
    'image': None
}


def load_text_model():
    import joblib
    if models_cache['text'] is None:
        try:
            models_cache['text'] = joblib.load(os.path.join(BASE_DIR, "emotion_model.pkl"))
            models_cache['text_vec'] = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
        except:
            return False
    return True


def load_voice_model():
    import joblib
    if models_cache['voice'] is None:
        try:
            models_cache['voice'] = joblib.load(os.path.join(BASE_DIR, "voicedata", "emotion_model.pkl"))
            models_cache['voice_le'] = joblib.load(os.path.join(BASE_DIR, "voicedata", "label_encoder.pkl"))
        except:
            return False
    return True


def load_image_model():
    if models_cache['image'] is None:
        try:
            import tensorflow as tf
            models_cache['image'] = tf.keras.models.load_model(os.path.join(BASE_DIR, "image", "emotion_model_final.h5"))
        except:
            return False
    return True


def make_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ══════════════════════════════════════════════════════════════════════════════
# TEXT EMOTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_text_pipeline(q: queue.Queue):
    try:
        q.put({"type": "progress", "stage": "init", "message": "📂 Loading Mental Health Dataset…", "percent": 5})
        time.sleep(0.5)

        # Lazy imports so they don't slow down startup
        import pandas as pd
        import re
        import string
        import nltk
        import joblib
        from nltk.corpus import stopwords
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report

        nltk.download("stopwords", quiet=True)

        dataset_path = os.path.join(BASE_DIR, "Mental Health Dataset.csv")
        df = pd.read_csv(dataset_path)

        q.put({"type": "progress", "stage": "load", "message": f"✅ Dataset loaded — {df.shape[0]:,} rows × {df.shape[1]} columns", "percent": 15})
        time.sleep(0.3)

        # Text cleaning
        q.put({"type": "progress", "stage": "preprocess", "message": "🔧 Cleaning and preprocessing text…", "percent": 25})
        stop_words = set(stopwords.words("english"))

        def clean_text(text):
            text = str(text).lower()
            text = text.translate(str.maketrans("", "", string.punctuation))
            words = text.split()
            words = [w for w in words if w not in stop_words]
            return " ".join(words)

        X = df["posts"].apply(clean_text)
        y = df["predicted"]

        q.put({"type": "progress", "stage": "preprocess", "message": "✅ Text preprocessing complete", "percent": 35})
        time.sleep(0.3)

        # TF-IDF
        q.put({"type": "progress", "stage": "features", "message": "🔢 Extracting TF-IDF features (max 5000)…", "percent": 45})
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        q.put({"type": "progress", "stage": "features", "message": f"✅ Features ready — train: {X_train_vec.shape[0]:,}, test: {X_test_vec.shape[0]:,}", "percent": 55})
        time.sleep(0.3)

        # Training
        q.put({"type": "progress", "stage": "training", "message": "🏋️ Training Logistic Regression model…", "percent": 65})
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_vec, y_train)

        q.put({"type": "progress", "stage": "training", "message": "✅ Training complete!", "percent": 75})
        time.sleep(0.3)

        # Testing
        q.put({"type": "progress", "stage": "testing", "message": "🧪 Running model evaluation on test set…", "percent": 82})
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        report_str = classification_report(y_test, y_pred, output_dict=True)

        q.put({"type": "progress", "stage": "testing", "message": f"✅ Evaluation done — Accuracy: {acc*100:.2f}%", "percent": 90})
        time.sleep(0.3)

        # Save model
        save_path = os.path.join(BASE_DIR, "emotion_model.pkl")
        vec_path = os.path.join(BASE_DIR, "vectorizer.pkl")
        joblib.dump(model, save_path)
        joblib.dump(vectorizer, vec_path)

        q.put({"type": "progress", "stage": "saving", "message": "💾 Model saved to emotion_model.pkl", "percent": 95})
        time.sleep(0.3)

        # Build per-class metrics
        class_metrics = []
        for label, metrics in report_str.items():
            if label in ("accuracy", "macro avg", "weighted avg"):
                continue
            if isinstance(metrics, dict):
                class_metrics.append({
                    "class": str(label),
                    "precision": round(metrics.get("precision", 0) * 100, 1),
                    "recall": round(metrics.get("recall", 0) * 100, 1),
                    "f1": round(metrics.get("f1-score", 0) * 100, 1),
                    "support": int(metrics.get("support", 0))
                })

        q.put({
            "type": "result",
            "modality": "text",
            "accuracy": round(acc * 100, 2),
            "macro_precision": round(report_str["macro avg"]["precision"] * 100, 2),
            "macro_recall": round(report_str["macro avg"]["recall"] * 100, 2),
            "macro_f1": round(report_str["macro avg"]["f1-score"] * 100, 2),
            "class_metrics": class_metrics,
            "summary": f"Logistic Regression trained on {X_train.shape[0]:,} samples, tested on {X_test.shape[0]:,} samples.",
            "percent": 100
        })

    except Exception as e:
        q.put({"type": "error", "message": f"❌ Text pipeline error: {str(e)}", "traceback": traceback.format_exc()})


# ══════════════════════════════════════════════════════════════════════════════
# VOICE EMOTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_voice_pipeline(q: queue.Queue):
    try:
        q.put({"type": "progress", "stage": "init", "message": "📂 Scanning voice dataset folders…", "percent": 5})
        time.sleep(0.5)

        import numpy as np
        import librosa
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score, classification_report
        import joblib

        DATA_PATH = os.path.join(BASE_DIR, "voicedata")

        # Scan folders first
        folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]
        total_files = sum(
            len([fn for fn in os.listdir(os.path.join(DATA_PATH, f)) if fn.endswith(".wav")])
            for f in folders
        )

        q.put({"type": "progress", "stage": "load", "message": f"✅ Found {len(folders)} emotion folders, {total_files:,} .wav files", "percent": 10})
        time.sleep(0.3)

        q.put({"type": "progress", "stage": "features", "message": "🎵 Extracting MFCC features from audio files…", "percent": 15})

        X = []
        y_labels = []
        processed = 0

        for folder in folders:
            folder_path = os.path.join(DATA_PATH, folder)
            if not os.path.isdir(folder_path):
                continue
            emotion = folder.split("_")[-1]
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(folder_path, file)
                    try:
                        audio, sr = librosa.load(file_path, duration=3, offset=0.5)
                        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                        X.append(np.mean(mfccs.T, axis=0))
                        y_labels.append(emotion)
                        processed += 1
                        if processed % 200 == 0:
                            pct = 15 + int((processed / max(total_files, 1)) * 45)
                            q.put({"type": "progress", "stage": "features",
                                   "message": f"🎵 Processed {processed:,}/{total_files:,} audio files…",
                                   "percent": min(pct, 60)})
                    except Exception:
                        pass

        X = np.array(X)
        y_labels = np.array(y_labels)

        q.put({"type": "progress", "stage": "features", "message": f"✅ Features extracted — shape: {X.shape}", "percent": 62})
        time.sleep(0.3)

        # Encode labels
        le = LabelEncoder()
        y_enc = le.fit_transform(y_labels)

        q.put({"type": "progress", "stage": "training", "message": "🏋️ Training Random Forest (200 estimators)…", "percent": 67})
        time.sleep(0.3)

        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=200, n_jobs=-1)
        model.fit(X_train, y_train)

        q.put({"type": "progress", "stage": "training", "message": "✅ Training complete!", "percent": 80})
        time.sleep(0.3)

        q.put({"type": "progress", "stage": "testing", "message": "🧪 Evaluating on test set…", "percent": 85})
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report_str = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

        q.put({"type": "progress", "stage": "testing", "message": f"✅ Evaluation done — Accuracy: {acc*100:.2f}%", "percent": 90})
        time.sleep(0.3)

        # Save
        save_path = os.path.join(BASE_DIR, "voicedata", "emotion_model.pkl")
        le_path = os.path.join(BASE_DIR, "voicedata", "label_encoder.pkl")
        joblib.dump(model, save_path)
        joblib.dump(le, le_path)

        q.put({"type": "progress", "stage": "saving", "message": "💾 Model saved to voicedata/emotion_model.pkl", "percent": 95})
        time.sleep(0.3)

        class_metrics = []
        for label, metrics in report_str.items():
            if label in ("accuracy", "macro avg", "weighted avg"):
                continue
            if isinstance(metrics, dict):
                class_metrics.append({
                    "class": str(label),
                    "precision": round(metrics.get("precision", 0) * 100, 1),
                    "recall": round(metrics.get("recall", 0) * 100, 1),
                    "f1": round(metrics.get("f1-score", 0) * 100, 1),
                    "support": int(metrics.get("support", 0))
                })

        q.put({
            "type": "result",
            "modality": "voice",
            "accuracy": round(acc * 100, 2),
            "macro_precision": round(report_str["macro avg"]["precision"] * 100, 2),
            "macro_recall": round(report_str["macro avg"]["recall"] * 100, 2),
            "macro_f1": round(report_str["macro avg"]["f1-score"] * 100, 2),
            "class_metrics": class_metrics,
            "summary": f"Random Forest trained on {X_train.shape[0]:,} audio samples, tested on {X_test.shape[0]:,} samples.",
            "percent": 100
        })

    except Exception as e:
        q.put({"type": "error", "message": f"❌ Voice pipeline error: {str(e)}", "traceback": traceback.format_exc()})


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE / FACIAL EXPRESSION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_image_pipeline(q: queue.Queue):
    try:
        q.put({"type": "progress", "stage": "init", "message": "📂 Setting up image data generators…", "percent": 5})
        time.sleep(0.5)

        import tensorflow as tf
        import numpy as np
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

        IMAGE_DIR = os.path.join(BASE_DIR, "image")
        TRAIN_DIR = os.path.join(IMAGE_DIR, "train")
        TEST_DIR = os.path.join(IMAGE_DIR, "test")

        img_size = 48
        batch_size = 32

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True
        )
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_data = train_datagen.flow_from_directory(
            TRAIN_DIR, target_size=(img_size, img_size),
            batch_size=batch_size, color_mode="grayscale", class_mode="categorical"
        )
        test_data = test_datagen.flow_from_directory(
            TEST_DIR, target_size=(img_size, img_size),
            batch_size=batch_size, color_mode="grayscale", class_mode="categorical"
        )

        num_classes = len(train_data.class_indices)
        class_names = list(train_data.class_indices.keys())

        q.put({"type": "progress", "stage": "load",
               "message": f"✅ Data loaded — {train_data.samples:,} train images, {test_data.samples:,} test images, {num_classes} classes",
               "percent": 15})
        time.sleep(0.3)

        q.put({"type": "progress", "stage": "model", "message": "🧠 Building CNN architecture…", "percent": 20})

        model = Sequential([
            Input(shape=(48, 48, 1)),
            Conv2D(32, (3, 3), activation='relu'), BatchNormalization(), MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'), BatchNormalization(), MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'), BatchNormalization(), MaxPooling2D(2, 2),
            Flatten(),
            Dense(256, activation='relu'), Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        q.put({"type": "progress", "stage": "model", "message": f"✅ CNN built — {model.count_params():,} parameters", "percent": 25})
        time.sleep(0.3)

        # Custom callback to stream epoch progress
        class ProgressCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                total_epochs = self.params.get("epochs", 10)
                pct = 25 + int(((epoch + 1) / total_epochs) * 55)
                q.put({
                    "type": "progress",
                    "stage": "training",
                    "message": (
                        f"📈 Epoch {epoch+1}/{total_epochs} — "
                        f"loss: {logs.get('loss', 0):.4f}, "
                        f"acc: {logs.get('accuracy', 0)*100:.2f}%, "
                        f"val_acc: {logs.get('val_accuracy', 0)*100:.2f}%"
                    ),
                    "percent": min(pct, 80),
                    "epoch_data": {
                        "epoch": epoch + 1,
                        "total": total_epochs,
                        "loss": round(logs.get("loss", 0), 4),
                        "accuracy": round(logs.get("accuracy", 0) * 100, 2),
                        "val_loss": round(logs.get("val_loss", 0), 4),
                        "val_accuracy": round(logs.get("val_accuracy", 0) * 100, 2)
                    }
                })

        q.put({"type": "progress", "stage": "training", "message": "🏋️ Starting CNN training (up to 30 epochs with early stopping)…", "percent": 26})

        save_path = os.path.join(IMAGE_DIR, "best_emotion_model.h5")
        final_path = os.path.join(IMAGE_DIR, "emotion_model_final.h5")

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True)

        history = model.fit(
            train_data, epochs=30,
            validation_data=test_data,
            callbacks=[early_stop, checkpoint, ProgressCallback()],
            verbose=0
        )
        epochs_ran = len(history.history['accuracy'])

        q.put({"type": "progress", "stage": "training", "message": f"✅ Training complete after {epochs_ran} epochs!", "percent": 82})
        time.sleep(0.3)

        # Evaluate
        q.put({"type": "progress", "stage": "testing", "message": "🧪 Evaluating on test set…", "percent": 85})
        loss, acc = model.evaluate(test_data, verbose=0)

        q.put({"type": "progress", "stage": "testing", "message": f"✅ Test accuracy: {acc*100:.2f}%", "percent": 90})
        time.sleep(0.3)

        model.save(final_path)
        q.put({"type": "progress", "stage": "saving", "message": "💾 Model saved to image/emotion_model_final.h5", "percent": 95})
        time.sleep(0.3)

        # Per-epoch history for charts
        epoch_history = []
        for i in range(epochs_ran):
            epoch_history.append({
                "epoch": i + 1,
                "accuracy": round(history.history['accuracy'][i] * 100, 2),
                "val_accuracy": round(history.history['val_accuracy'][i] * 100, 2),
                "loss": round(history.history['loss'][i], 4),
                "val_loss": round(history.history['val_loss'][i], 4)
            })

        q.put({
            "type": "result",
            "modality": "image",
            "accuracy": round(acc * 100, 2),
            "loss": round(loss, 4),
            "epochs_ran": epochs_ran,
            "num_classes": num_classes,
            "class_names": class_names,
            "epoch_history": epoch_history,
            "train_samples": train_data.samples,
            "test_samples": test_data.samples,
            "summary": f"CNN trained on {train_data.samples:,} facial images across {num_classes} emotion classes for {epochs_ran} epochs.",
            "percent": 100
        })

    except Exception as e:
        q.put({"type": "error", "message": f"❌ Image pipeline error: {str(e)}", "traceback": traceback.format_exc()})


# ══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/run/<modality>", methods=["POST"])
def run_pipeline(modality: str):
    if modality not in ("text", "voice", "image"):
        return jsonify({"error": "Invalid modality"}), 400

    job_id = f"{modality}_{int(time.time())}"
    q: queue.Queue = queue.Queue()

    with job_lock:
        job_queues[job_id] = q

    pipeline_map = {
        "text": run_text_pipeline,
        "voice": run_voice_pipeline,
        "image": run_image_pipeline,
    }

    thread = threading.Thread(target=pipeline_map[modality], args=(q,), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/stream/<job_id>")
def stream_job(job_id: str):
    def generate():
        with job_lock:
            q = job_queues.get(job_id)

        if q is None:
            yield make_event({"type": "error", "message": "Job not found"})
            return

        while True:
            try:
                event = q.get(timeout=120)
                yield make_event(event)
                if event.get("type") in ("result", "error"):
                    break
            except queue.Empty:
                yield make_event({"type": "heartbeat"})

        with job_lock:
            job_queues.pop(job_id, None)

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "models": ["text", "voice", "image"]})


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/detect/text", methods=["POST"])
def detect_text():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    if not load_text_model():
        return jsonify({"error": "Text model not found. Please run the pipeline first."}), 404

    import string
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))

    def clean_text(t):
        t = str(t).lower()
        t = t.translate(str.maketrans("", "", string.punctuation))
        words = t.split()
        words = [w for w in words if w not in stop_words]
        return " ".join(words)

    cleaned = clean_text(text)
    vec = models_cache['text_vec'].transform([cleaned])
    pred = models_cache['text'].predict(vec)

    return jsonify({"emotion": str(pred[0])})


@app.route("/api/detect/voice", methods=["POST"])
def detect_voice():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not load_voice_model():
        return jsonify({"error": "Voice model not found. Please run the pipeline first."}), 404

    import librosa
    import numpy as np
    import tempfile

    # Detect the real content type — browsers record webm/opus regardless of filename
    orig_name = file.filename or "recording.webm"
    content_type = file.content_type or ""

    # Determine extension from content type first, then filename
    if "webm" in content_type or "webm" in orig_name:
        ext = ".webm"
    elif "ogg" in content_type or "ogg" in orig_name:
        ext = ".ogg"
    elif "mp4" in content_type or "mp4" in orig_name:
        ext = ".mp4"
    elif "wav" in content_type or orig_name.endswith(".wav"):
        ext = ".wav"
    else:
        ext = ".webm"  # browser default

    temp_path = None
    wav_path = None
    try:
        # Save uploaded file with CORRECT extension
        tmp_fd, temp_path = tempfile.mkstemp(suffix=ext, dir=BASE_DIR)
        os.close(tmp_fd)
        file.save(temp_path)

        # Always convert non-wav to wav via pydub (requires ffmpeg)
        if ext.lower() != ".wav":
            try:
                from pydub import AudioSegment
                audio_seg = AudioSegment.from_file(temp_path)
                wav_fd, wav_path = tempfile.mkstemp(suffix=".wav", dir=BASE_DIR)
                os.close(wav_fd)
                audio_seg.export(wav_path, format="wav")
                load_path = wav_path
            except Exception as conv_err:
                print(f"pydub conversion failed ({conv_err}), trying librosa direct load")
                load_path = temp_path
        else:
            load_path = temp_path

        # Load audio — use full duration (no offset) so short recordings still work
        audio, sr = librosa.load(load_path, sr=22050, mono=True)

        # Pad or trim to exactly 3 seconds so MFCC shape is consistent
        target_len = sr * 3
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
        else:
            # Skip a small silence offset if longer than 3s
            offset_samples = int(0.5 * sr)
            audio = audio[offset_samples:offset_samples + target_len]

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        features = np.mean(mfccs.T, axis=0).reshape(1, -1)

        pred = models_cache['voice'].predict(features)
        emotion = models_cache['voice_le'].inverse_transform(pred)

        # Get confidence if the model supports predict_proba
        confidence = None
        all_scores = {}
        try:
            proba = models_cache['voice'].predict_proba(features)[0]
            classes = models_cache['voice_le'].classes_
            all_scores = {str(classes[i]): round(float(proba[i]) * 100, 1) for i in range(len(classes))}
            confidence = round(float(np.max(proba)) * 100, 1)
        except Exception:
            pass

        return jsonify({
            "emotion": str(emotion[0]),
            "confidence": confidence,
            "all_scores": all_scores
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Voice detection failed: {str(e)}"}), 500
    finally:
        for p in [temp_path, wav_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass


@app.route("/api/detect/image", methods=["POST"])
def detect_image():
    import cv2
    import numpy as np

    if 'file' in request.files:
        file_bytes = request.files['file'].read()
    elif request.json and 'image' in request.json:
        import base64
        file_bytes = base64.b64decode(request.json['image'].split(",")[-1])
    else:
        return jsonify({"error": "No image provided"}), 400

    if not load_image_model():
        return jsonify({"error": "Image model not found. Please run the pipeline first."}), 404

    nparr = np.frombuffer(file_bytes, np.uint8)

    # Decode as color first for face detection
    frame_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame_color is None:
        return jsonify({"error": "Invalid image — could not decode."}), 400

    gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

    # ── Face detection with Haar cascade ────────────────────────────────────
    # Use OpenCV's built-in frontalface cascade (ships with cv2)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(20, 20)
    )

    face_region = None
    face_info = "full image (no face detected — using center crop)"

    if len(faces) > 0:
        # Pick the largest detected face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        # Add a small padding around the face (10%)
        pad = int(0.10 * min(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad)
        y2 = min(gray.shape[0], y + h + pad)
        face_region = gray[y1:y2, x1:x2]
        face_info = f"face detected at ({x},{y}) size {w}x{h}"
    else:
        # Fallback: center-crop the image (assume face is roughly centered)
        h_img, w_img = gray.shape
        crop_size = min(h_img, w_img)
        y_start = (h_img - crop_size) // 2
        x_start = (w_img - crop_size) // 2
        face_region = gray[y_start:y_start + crop_size, x_start:x_start + crop_size]

    # Resize to the exact input size the model was trained on
    face_img = cv2.resize(face_region, (48, 48))

    # Apply histogram equalization to improve contrast (matches training preprocessing)
    face_img = cv2.equalizeHist(face_img)

    face_img = face_img.astype(np.float32) / 255.0
    face_img = np.reshape(face_img, (1, 48, 48, 1))

    pred = models_cache['image'].predict(face_img)
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    idx = int(np.argmax(pred))
    emotion = emotion_labels[idx]
    confidence = float(np.max(pred)) * 100

    # Build per-emotion confidence breakdown
    all_scores = {
        emotion_labels[i]: round(float(pred[0][i]) * 100, 1)
        for i in range(len(emotion_labels))
    }

    return jsonify({
        "emotion": emotion,
        "confidence": round(confidence, 1),
        "all_scores": all_scores,
        "face_info": face_info
    })


if __name__ == "__main__":
    app.run(debug=True, port=5050, threaded=True)
