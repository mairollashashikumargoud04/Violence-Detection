from django.shortcuts import render,redirect
import pymysql
from django.http import HttpResponse
from Database import getConnection
from django.conf import settings
from django.contrib import messages
import os
import numpy as np
import cv2
from tqdm import tqdm
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tempfile
from django.http import JsonResponse
# Keras / TF (ensure tensorflow is installed)
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from sklearn.model_selection import train_test_split
import winsound


# Create your views here.

def index(request):
    return render(request,'index.html')

def ulogin(request):
    return render(request,'user/Login.html')

def register(request):
    return render(request,'user/Register.html')

def RegAction(request):
    con = getConnection()
    cur = con.cursor()

    try:
        n = request.POST.get('name')
        e = request.POST.get('email')
        u = request.POST.get('username')
        p = request.POST.get('password')

        # Connect to MySQL

        # ✅ Check if email already exists
        check_query = "SELECT * FROM user WHERE email=%s"
        cur.execute(check_query, (e,))
        existing_user = cur.fetchone()

        if existing_user:
            context={'msg':"Email already registered. Please use another email or login."}
            return render(request,'user/Register.html',context)

        # Insert user into table "users"
        query = "INSERT INTO user(name, email, username, password) VALUES (%s, %s, %s, %s)"
        cur.execute(query, (n, e, u, p))
        con.commit()

        context={'msg':"Registration successful! Please login."}
        return render(request, 'user/Login.html',context)  # make sure you have a login route/view

    except Exception as ex:
        print("Error:", ex)
        messages.error(request, "Something went wrong while registering!")

    finally:
        cur.close()
        con.close()

    return render(request, "user/Register.html")

def logaction(request):
        e = request.POST.get('email')
        p = request.POST.get('password')

        con = getConnection()
        cur = con.cursor()


        query = "SELECT * FROM user WHERE email=%s AND password=%s"
        cur.execute(query, (e, p))
        user = cur.fetchone()

        if user:
            request.session['email'] = e
            return render(request,'user/UserHome.html')
        else:
            context = {'msg': "Invalid email or password!"}
            return render(request, 'user/Login.html', context)


def userhome(request):
    return render(request,'user/UserHome.html')

def alogin(request):
    return render(request,'admin/Login.html')

def alogaction(request):
    u = request.POST.get('username')
    p = request.POST.get('password')

    if u == 'Admin' and p == 'Admin':
        return render(request,'admin/Home.html')
    else:
        context = {'msg': "Invalid Username or password!"}
        return render(request, 'admin/Login.html', context)

def admin_home(request):
    return render(request, 'admin/Home.html')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, 'media', 'features_dl.npz')  # will store per-frame features
BEST_MODEL_INFO = os.path.join(BASE_DIR, 'Model', 'best_model.json')


def admin_upload_dataset(request):
    if request.method == 'POST':
        # Check files
        files = request.FILES.getlist('dataset_dir')
        if not files:
            return redirect(request,'admin/Home.html',{'msg':"No files received. Please select a folder."})

        # Destination base path
        base_dir = os.path.join(settings.MEDIA_ROOT, 'dataset')
        os.makedirs(base_dir, exist_ok=True)

        saved_count = 0
        for f in files:
            # Each uploaded file has webkitRelativePath like "Violence/video1.mp4"
            rel_path = getattr(f, 'webkitRelativePath', None) or f.name
            # Ensure safe path (no .. etc)
            rel_path = rel_path.replace('\\', '/').lstrip('/')
            dest_path = os.path.join(base_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # Save file
            with open(dest_path, 'wb+') as dest:
                for chunk in f.chunks():
                    dest.write(chunk)
            saved_count += 1

        return render(request,'admin/Home.html',{'msg':f"✅ Dataset uploaded successfully! {saved_count} files saved to {base_dir}"})

        # If not POST
    return render(request, 'admin/Home.html',{'msg':"Invalid request method."})

def frames_from_video(video_path, num_frames=16, size=(224,224)):
    """
    Read video and return num_frames frames uniformly sampled resized to size.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total-1,0), num_frames, dtype=int)
    frames = []
    for idx in indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()

        if not ret:
            if len(frames)>0:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((size[1], size[0],3), dtype=np.uint8))
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, size)
            frames.append(frame)
    cap.release()
    return np.array(frames)

def compute_video_features(frames):
    """
    Given frames (num_frames, h, w, 3), compute:
      - motion_score: mean absolute difference between consecutive frames (RGB -> grayscale)
      - edge_score: mean Canny edge count per frame (normalized)
      - brightness: mean intensity
    Returns a dict of features.
    """
    # convert to gray
    gray = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames], dtype=np.float32)
    # motion: mean absolute diff between consecutive frames
    diffs = np.abs(np.diff(gray, axis=0))
    motion_per_frame = diffs.mean(axis=(1,2))  # per-frame motion intensity
    motion_score = float(motion_per_frame.mean())  # average motion over clip

    # edge: use Canny and count edges
    edge_counts = []
    for g in gray:
        edges = cv2.Canny(g.astype(np.uint8), 100, 200)
        edge_counts.append(edges.sum() / (edges.shape[0]*edges.shape[1]))  # normalized
    edge_score = float(np.mean(edge_counts))

    brightness = float(gray.mean()/255.0)  # normalized 0..1

    return {'motion_score': motion_score, 'edge_score': edge_score, 'brightness': brightness}

def preprocess_dataset(dataset_dir, frames=16, size=(224,224)):
    """
    Walk dataset_dir expecting 'violence' and 'non_violence', compute features for each video and save to FEATURES_PATH.
    """
    classes = ['NonViolence', 'Violence']
    videos = []
    labels = []
    for label_idx, cls in enumerate(classes):
        class_dir = os.path.join(dataset_dir, cls)
        if not os.path.exists(class_dir):
            print(f"Warning: missing {class_dir}")
            continue
        for fn in os.listdir(class_dir):
            if fn.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                videos.append(os.path.join(class_dir, fn))
                labels.append(label_idx)

    if len(videos) == 0:
        raise RuntimeError("No videos found for DL preprocessing.")

    # MobileNetV2 backbone to get per-frame features
    backbone = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(size[1], size[0], 3))
    all_feats = []
    valid_labels = []

    print("Extracting per-frame features using MobileNetV2 ...")
    for vid, lab in tqdm(zip(videos, labels), total=len(videos)):
        try:
            frames_arr = frames_from_video(vid, num_frames=frames, size=size)  # uses function above
        except Exception as e:
            print("Skip video", vid, ":", e)
            continue
        # preprocess and extract features
        x = mobilenet_preprocess(frames_arr.astype('float32'))
        feats = backbone.predict(x, verbose=0)  # (frames, feat_dim)
        all_feats.append(feats)
        valid_labels.append(lab)

    X = np.array(all_feats)  # (n, frames, feat_dim)
    y = np.array(valid_labels)
    np.savez(FEATURES_PATH, X=X, y=y)
    print("Saved DL features to", FEATURES_PATH)
    return {'n_videos': X.shape[0], 'frames': X.shape[1], 'feat_dim': X.shape[2]}


def admin_process_videos(request):
    if request.method != 'POST':
        return render(request,'admin/Home.html',{'message':"Invalid request method for preprocess. Please use the Preprocess button."})

    frame = int(request.POST.get('frames', 16))  # default 16
    size = int(request.POST.get('size', 224))  # de
    split = request.POST.get('split')

        # change this to the actual folder name you have inside media/dataset/
    dataset_dir = 'Real_Life_Violence_Dataset'

    try:
        info = preprocess_dataset(dataset_dir, frames=frame, size=(size, size))
        n = info.get('n_videos', 0)
        # FEATURES_PATH is saved location from training.py
        return render(request, 'admin/Home.html',{'message':f"✅ Preprocessing completed. Processed {n} videos. Features saved to {FEATURES_PATH}"})
    except Exception as e:
        # capture exception message and return as error
        return render(request, 'admin/Home.html',{'message':f"Preprocess failed: {str(e)}"})
    return render(request,'admin/Home.html')

# Model builders
def build_avg_dense(input_shape):
    inp = Input(shape=input_shape)  # (frames, feat)
    x = layers.Lambda(lambda z: tf.reduce_mean(z, axis=1))(inp)  # average pooling across time
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm(input_shape):
    inp = Input(shape=input_shape)
    x = layers.Masking()(inp)
    x = layers.LSTM(128)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

PLOT_PATH = os.path.join(BASE_DIR, 'Static', 'images', 'performance.png')
def admin_train_model(request):
    epochs =int(request.POST['epochs'])
    batch_size = int(request.POST['batch'])
    model_dir=None
    if model_dir is None:
        model_dir = os.path.join(BASE_DIR, 'Model')
    os.makedirs(model_dir, exist_ok=True)

    data = np.load("media/features_dl.npz")
    X = data['X']  # shape (n, frames, feat_dim)
    y = data['y']

    # Split into train and test (we'll keep a holdout test for final eval)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    input_shape = X_train.shape[1:]  # (frames, feat_dim)
    histories = {}
    saved_models = {}

    # AVG model
    avg_model_path = os.path.join(model_dir, 'model_avg.h5')
    # if not os.path.exists(avg_model_path):
    m_avg = build_avg_dense(input_shape)
    h_avg = m_avg.fit(X_train, y_train, validation_data=(X_test, y_test),
                          epochs=epochs, batch_size=batch_size, verbose=1)
    m_avg.save(avg_model_path)
    np.savez(os.path.join(model_dir, 'history_avg.npz'),
                 history=h_avg.history)
    histories['avg'] = h_avg.history
    saved_models['avg'] = avg_model_path
    # else:
    #     return render(request, 'admin/Home.html',{'train_msg':"Avg model exists, skipping training."})
    # LSTM model
    lstm_model_path = os.path.join(model_dir, 'model_lstm.h5')
    # if not os.path.exists(lstm_model_path):
    m_lstm = build_lstm(input_shape)
    h_lstm = m_lstm.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=epochs, batch_size=batch_size, verbose=1)
    m_lstm.save(lstm_model_path)
    np.savez(os.path.join(model_dir, 'history_lstm.npz'),
                 history=h_lstm.history)
    histories['lstm'] = h_lstm.history
    saved_models['lstm'] = lstm_model_path
    # else:
    #     return render(request, 'admin/Home.html', {'train_msg': "LSTM model exists, skipping training."})

    # Evaluate to pick best model based on validation accuracy logged in history
    # For models just trained we check their last val_accuracy. For pre-existing we try to load history files.
    val_acc = {}
    for mname in ['avg', 'lstm']:
        hist_file = os.path.join(model_dir, f'history_{mname}.npz')
        if os.path.exists(hist_file):
            hist = np.load(hist_file, allow_pickle=True)['history'].item()
            # prefer 'val_accuracy' key or 'val_acc' older TF
            va = hist.get('val_accuracy') or hist.get('val_acc') or [0.0]
            val_acc[mname] = va[-1] if isinstance(va, (list, np.ndarray)) else float(va)
        else:
            val_acc[mname] = 0.0

    # pick best
    best_name = max(val_acc, key=val_acc.get)
    best_model_file = os.path.join(model_dir, f"model_{best_name}.h5")
    with open(BEST_MODEL_INFO, 'w') as f:
        json.dump({'best': os.path.basename(best_model_file), 'val_acc': float(val_acc[best_name])}, f)

    # create a combined performance plot (train/val accuracy for both if available)
    try:

        plt.figure(figsize=(8,5))
        for mname in ['avg','lstm']:
            hist_file = os.path.join(model_dir, f'history_{mname}.npz')
            if os.path.exists(hist_file):
                hist = np.load(hist_file, allow_pickle=True)['history'].item()
                if 'accuracy' in hist:
                    plt.plot(hist['accuracy'], label=f"{mname} train acc")
                if 'val_accuracy' in hist:
                    plt.plot(hist['val_accuracy'], linestyle='--', label=f"{mname} val acc")
        plt.title("Model training & validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOT_PATH)
        plt.close()
    except Exception as e:
        print("Plotting failed:", e)

    return render(request,'admin/Home.html',{'val_acc': val_acc, 'best': best_name})

def predict_video_dl(video_path, frames=16, size=(224,224)):
    import tensorflow as tf
    """
    Play video and detect violence. If violence detected, draw rectangle and beep.
    Fixes window size and text scaling so the video and text are readable.
    """
    import json, time, math
    # load model info
    if not os.path.exists(BEST_MODEL_INFO):
        raise RuntimeError("Best model info missing. Train first.")
    with open(BEST_MODEL_INFO, 'r') as f:
        info = json.load(f)
    best_model_name = info.get('best')
    model_path = os.path.join(BASE_DIR, 'Model', best_model_name)
    if not os.path.exists(model_path):
        raise RuntimeError("Model file not found: " + model_path)

    model = tf.keras.models.load_model(model_path)
    backbone = MobileNetV2(weights='imagenet', include_top=False, pooling='avg',
                           input_shape=(size[1], size[0], 3))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + video_path)

    # get original video size and fps
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay_ms = int(1000 / fps)

    # create resizable window and set it to a reasonable size (scale down if very large)
    window_name = "Violence Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # compute display size (limit max width for screen fit)
    max_display_w = 960
    scale = 1.0
    if frame_w > max_display_w:
        scale = max_display_w / frame_w
    display_w = int(frame_w * scale)
    display_h = int(frame_h * scale)
    cv2.resizeWindow(window_name, display_w, display_h)

    violence_detected_anywhere = False
    last_prob = 0.0

    # For sliding-window style detection we'll move through video in steps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # We'll process in chunks centred around current frame for simpler code
    current_frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # keep a copy for drawing; frame_bgr is what we display
        display_frame = frame_bgr.copy()

        # prepare frames for model: sample 'frames' frames around current position
        # compute start index for sampling
        start_idx = max(0, current_frame_idx - frames // 2)
        # gather frames_arr in RGB (model expects RGB)
        frames_arr = []
        cap_pos_saved = cap.get(cv2.CAP_PROP_POS_FRAMES)  # remember
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        for i in range(frames):
            ret2, f = cap.read()
            if not ret2:
                break
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, size)
            frames_arr.append(rgb)
        # restore capture position to just after the current frame to continue playback
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap_pos_saved)

        # if we got enough frames, make prediction
        if len(frames_arr) > 0:
            x = mobilenet_preprocess(np.array(frames_arr, dtype='float32'))
            feats = backbone.predict(x, verbose=0)  # (frames, feat_dim)
            X = np.expand_dims(feats, axis=0)
            p = float(model.predict(X, verbose=0)[0][0])
            last_prob = p
            label = 1 if p >= 0.5 else 0
        else:
            p = last_prob
            label = 1 if p >= 0.5 else 0

        # Choose font size & thickness relative to frame height so it scales
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.6, frame_h / 480.0)   # smaller videos -> ~0.6, larger -> scaled up
        thickness = max(1, int(frame_h / 400))   # thickness scales with resolution
        # text position
        text = f"{'VIOLENCE' if label==1 else 'NON-VIOLENCE'}"
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = 10
        text_y = 30 + text_size[1]

        if label == 1:
            violence_detected_anywhere = True
            # rectangle coordinates — make it relative to frame size
            # Here we draw a large rectangle inset by 5% margin
            margin_x = int(frame_w * 0.05)
            margin_y = int(frame_h * 0.05)
            top_left = (margin_x, margin_y)
            bottom_right = (frame_w - margin_x, frame_h - margin_y)
            # red rectangle (BGR)
            cv2.rectangle(display_frame, top_left, bottom_right, (0, 0, 255), max(2, int(frame_w/200)))
            cv2.putText(display_frame, text, (text_x, text_y), font, font_scale, (0,0,255), thickness, cv2.LINE_AA)

            # beep: on Windows use winsound, else fallback to simple cross-platform beep
            try:
                import winsound
                winsound.Beep(1000, 180)   # freq=1000Hz, duration=180ms
            except Exception:
                # fallback: try print('\a') which may trigger terminal beep on some systems
                print('\a', end='', flush=True)
        else:
            # green text for non-violence
            cv2.putText(display_frame, text, (text_x, text_y), font, font_scale, (0,255,0), thickness, cv2.LINE_AA)

        # Resize display_frame to display_w/display_h for window if we scaled earlier
        if scale != 1.0:
            disp = cv2.resize(display_frame, (display_w, display_h))
        else:
            disp = display_frame

        cv2.imshow(window_name, disp)

        # wait depending on fps (use delay_ms)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord('q'):
            break

        current_frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    return {
        'label_name': 'violence' if violence_detected_anywhere else 'non_violence',
        'probability': round(float(last_prob), 3),
        'label': 1 if violence_detected_anywhere else 0
    }
def user_upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        # save temp file
        tmpdir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmpdir, video_file.name)
        with open(tmp_path, 'wb') as f:
            for chunk in video_file.chunks():
                f.write(chunk)
        try:
            res = predict_video_dl(tmp_path, frames=16, size=(224, 224))
            return render(request,'user/UserHome.html',{'result': res})

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    return JsonResponse({'status': 'error', 'message': 'send POST with "video" file'})