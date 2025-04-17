from flask import Flask, render_template, Response, redirect, url_for, request, session, flash
from flask import jsonify
from flask_bcrypt import Bcrypt
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import pygame
import threading
import time
import sqlite3
import os
import secrets
import pytz
from queue import Queue
from datetime import datetime, timedelta

app = Flask(__name__)
bcrypt = Bcrypt(app)
app.secret_key = secrets.token_hex(32)  # Secure random secret key

# Initialize pygame for audio
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav")

#Global variables
camera_lock = threading.Lock()
alarm_queue = Queue(maxsize=1)  # Only allow one alarm at a time
camera = cv2.VideoCapture(0)  # Initialize camera once globally

# Detection variables
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 20
YAWN_THRESH = 0.6            # Raised from 0.5 so normal talking isn't flagged as yawning
COUNTER = 0
ALARM_ON = False
YAWNING = False

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Database setup
def init_db():
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()
    
    # Create tables with explicit error handling
    try:
        c.execute('''CREATE TABLE IF NOT EXISTS users
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS events
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp DATETIME)''')
        conn.commit()
        print("✅ Database tables verified/created")
    except sqlite3.Error as e:
        print(f"🚨 Database creation error: {e}")
    finally:
        conn.close()

init_db()  # Call this immediately after definition

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[0], mouth[6])
    B = distance.euclidean(mouth[2], mouth[10])
    C = distance.euclidean(mouth[4], mouth[8])
    mar = (B + C) / (2.0 * A)
    return mar

def log_event(user_id, event_type):
    print(f"🔔 Attempting to log: User {user_id}, Event: {event_type}")  # Debug
    
    try:
        # Create a timestamp in your timezone (Africa/Nairobi)
        tz = pytz.timezone("Africa/Nairobi")
        timestamp = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # Verify the user exists
        c.execute("SELECT id FROM users WHERE id=?", (user_id,))
        if not c.fetchone():
            print(f"❌ User {user_id} doesn't exist!")
            return False
        
        # Insert the event with our timezone-adjusted timestamp
        c.execute("INSERT INTO events (user_id, event_type, timestamp) VALUES (?, ?, ?)", 
                 (user_id, event_type, timestamp))
        conn.commit()
        print(f"✅ Successfully logged event for User {user_id}")
        return True
        
    except sqlite3.Error as e:
        print(f"🚨 Database Error: {str(e)}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def play_alarm():
    try:
        alarm_sound.play()
        time.sleep(3)  # Alarm duration
    except Exception as e:
        print(f"Alarm error: {e}")
    finally:
        alarm_queue.get()  # Release the alarm slot

def trigger_alarm():
    if alarm_queue.empty():  # Only trigger if no alarm is playing
        alarm_queue.put(1)
        threading.Thread(target=play_alarm, daemon=True).start()

# Cooldown settings for event logging
last_log_time = {"DROWSINESS": 0, "YAWNING": 0}
LOG_COOLDOWN = 10  # seconds

def detect_drowsiness_and_yawn(frame, user_id):
    global COUNTER, ALARM_ON, YAWNING
    
    # Early return if no frame
    if frame is None:
        return False, False, frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    # No faces detected
    if not rects:
        COUNTER = 0  # Reset if face disappears
        return False, False, frame

    shape = predictor(gray, rects[0])
    shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

    # Calculate EAR only if eyes are detectable
    try:
        left_ear = eye_aspect_ratio(shape[42:48])
        right_ear = eye_aspect_ratio(shape[36:42])
        ear = (left_ear + right_ear) / 2.0
    except:
        ear = 0.5  # Default open eyes value

    # Calculate MAR only if mouth is detectable
    try:
        mar = mouth_aspect_ratio(shape[48:68])
    except:
        mar = 0.3  # Default closed mouth value

    # Debug print actual values
    print(f"📊 EAR: {ear:.2f} | MAR: {mar:.2f} | Thresholds: {EYE_AR_THRESH}/{YAWN_THRESH}")

    # Strict logging conditions
    drowsy = ear < EYE_AR_THRESH
    yawning = mar > YAWN_THRESH

    current_time = time.time()
    # Only log state changes for drowsiness with cooldown check
    if drowsy:
        COUNTER += 1
        if COUNTER >= EYE_AR_CONSEC_FRAMES and not ALARM_ON:
            ALARM_ON = True
            trigger_alarm()
            if current_time - last_log_time["DROWSINESS"] >= LOG_COOLDOWN:
                log_event(user_id, "Drowsiness Detected")
                last_log_time["DROWSINESS"] = current_time
    else:
        COUNTER = 0
        ALARM_ON = False

    # Only log state changes for yawning with cooldown check
    if yawning and not YAWNING:
        YAWNING = True
        if current_time - last_log_time["YAWNING"] >= LOG_COOLDOWN:
            log_event(user_id, "Yawning Detected")
            last_log_time["YAWNING"] = current_time
    elif not yawning and YAWNING:
        YAWNING = False

    return drowsy, yawning, frame

def generate_frames(user_id):
    global camera

    while True:
        with camera_lock:
            success, frame = camera.read()
            if not success:
                # Attempt to reconnect camera
                camera.release()
                time.sleep(0.5)
                camera = cv2.VideoCapture(0)
                continue

        try:
            # Call your detection function with the correct arguments
            drowsy, yawning, frame = detect_drowsiness_and_yawn(frame, user_id)

            # Overlay messages on the frame based on detection
            if drowsy:
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if yawning:
                cv2.putText(frame, "YAWNING DETECTED", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Encode the frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            # Yield the frame to be displayed in the browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            print(f"Frame processing error: {e}")
            continue
      

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if not username or not password:
            flash('Please enter both username and password', 'error')
            return redirect(url_for('register'))
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                     (username, hashed_password))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists', 'error')
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT id, password FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        
        if user and bcrypt.check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/force_log')
def force_log():
    if 'user_id' not in session:
        return "Please login first"
    
    success = log_event(session['user_id'], "TEST_EVENT_FORCE")
    return f"Test event {'succeeded!' if success else 'failed'}"

@app.route('/debug_db')
def debug_db():
    """Check database status"""
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        tables = c.execute("SELECT name FROM sqlite_master").fetchall()
        events = c.execute("SELECT * FROM events").fetchall()
        users = c.execute("SELECT * FROM users").fetchall()
        
        return {
            "tables": tables,
            "users": users,
            "events": events,
            "your_session": dict(session)
        }
    except Exception as e:
        return {"error": str(e)}

@app.route('/simulate_event')
def simulate_event():
    """Manually create an event"""
    if 'user_id' not in session:
        return {"error": "Not logged in"}
    
    log_event(session['user_id'], "MANUAL_TEST_EVENT")
    return {"status": "Event logged"}

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get real events from database
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''SELECT event_type, timestamp FROM events 
                 WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5''',
              (session['user_id'],))
    events = c.fetchall()
    conn.close()
    
    # Format events for template and count event types for dashboard graph
    recent_events = []
    drowsy_count = 0
    yawning_count = 0
    for event in events:
        recent_events.append({
            'type': event[0],
            'time': event[1]  # Timestamp already adjusted at insert
        })
        if event[0] == "Drowsiness Detected":
            drowsy_count += 1
        elif event[0] == "Yawning Detected":
            yawning_count += 1
    
    return render_template('dashboard.html',
                         username=session['username'],
                         recent_events=recent_events,
                         drowsy_count=drowsy_count,
                         yawning_count=yawning_count)
    
@app.route('/dashboard_events')
def dashboard_events():
    if 'user_id' not in session:
        return ""
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''SELECT event_type, timestamp FROM events 
                 WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5''',
              (session['user_id'],))
    events = c.fetchall()
    conn.close()
    
    return render_template('events_table.html', events=events)    

@app.route('/video_feed')
def video_feed():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return Response(generate_frames(session['user_id']),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_alerts')
def check_alerts():
    if 'user_id' not in session:
        return {'error': 'Unauthorized'}, 401
    return {'drowsy': ALARM_ON, 'yawning': YAWNING}

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create register.html and login.html if they don't exist
    auth_templates = {
        'register.html': '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Register</title>
            <style>
                /* Same styling as previously provided for register page */
                body { font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }
                .auth-container { background-color: white; border-radius: 10px; box-shadow: 0 10px 20px rgba(0,0,0,0.1); padding: 2rem; width: 350px; }
                h1 { color: #4361ee; text-align: center; margin-bottom: 1.5rem; }
                .form-group { margin-bottom: 1.5rem; }
                label { display: block; margin-bottom: 0.5rem; color: #495057; }
                input { width: 100%; padding: 0.75rem; border: 1px solid #ced4da; border-radius: 4px; font-size: 1rem; }
                button { width: 100%; padding: 0.75rem; background-color: #4361ee; color: white; border: none; border-radius: 4px; font-size: 1rem; cursor: pointer; transition: background-color 0.3s; }
                button:hover { background-color: #3f37c9; }
                .message { padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem; text-align: center; }
                .error { background-color: #ffe3e3; color: #f03e3e; }
                .success { background-color: #d3f9d8; color: #2b8a3e; }
                .login-link { text-align: center; margin-top: 1rem; }
                a { color: #4361ee; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="auth-container">
                <h1>Create Account</h1>
                {% with messages = get_flashed_messages(with_categories=true) %}
                    {% for category, message in messages %}
                        <div class="message {{ category }}">{{ message }}</div>
                    {% endfor %}
                {% endwith %}
                <form method="POST">
                    <div class="form-group">
                        <label for="username">Username</label>
                        <input type="text" id="username" name="username" required>
                    </div>
                    <div class="form-group">
                        <label for="password">Password</label>
                        <input type="password" id="password" name="password" required>
                    </div>
                    <button type="submit">Register</button>
                </form>
                <div class="login-link">
                    Already have an account? <a href="{{ url_for('login') }}">Login</a>
                </div>
            </div>
        </body>
        </html>
        ''',
        'login.html': '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Login</title>
            <style>
                /* Same styling as previously provided for login page */
                body { font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }
                .auth-container { background-color: white; border-radius: 10px; box-shadow: 0 10px 20px rgba(0,0,0,0.1); padding: 2rem; width: 350px; }
                h1 { color: #4361ee; text-align: center; margin-bottom: 1.5rem; }
                .form-group { margin-bottom: 1.5rem; }
                label { display: block; margin-bottom: 0.5rem; color: #495057; }
                input { width: 100%; padding: 0.75rem; border: 1px solid #ced4da; border-radius: 4px; font-size: 1rem; }
                button { width: 100%; padding: 0.75rem; background-color: #4361ee; color: white; border: none; border-radius: 4px; font-size: 1rem; cursor: pointer; transition: background-color 0.3s; }
                button:hover { background-color: #3f37c9; }
                .message { padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem; text-align: center; }
                .error { background-color: #ffe3e3; color: #f03e3e; }
                .success { background-color: #d3f9d8; color: #2b8a3e; }
                .register-link { text-align: center; margin-top: 1rem; }
                a { color: #4361ee; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="auth-container">
                <h1>Login</h1>
                {% with messages = get_flashed_messages(with_categories=true) %}
                    {% for category, message in messages %}
                        <div class="message {{ category }}">{{ message }}</div>
                    {% endfor %}
                {% endwith %}
                <form method="POST">
                    <div class="form-group">
                        <label for="username">Username</label>
                        <input type="text" id="username" name="username" required>
                    </div>
                    <div class="form-group">
                        <label for="password">Password</label>
                        <input type="password" id="password" name="password" required>
                    </div>
                    <button type="submit">Login</button>
                </form>
                <div class="register-link">
                    Don't have an account? <a href="{{ url_for('register') }}">Register</a>
                </div>
            </div>
        </body>
        </html>
        '''
    }
    
    for template_name, template_content in auth_templates.items():
        template_path = os.path.join('templates', template_name)
        if not os.path.exists(template_path):
            with open(template_path, 'w') as f:
                f.write(template_content)
    
    app.run(debug=True)
