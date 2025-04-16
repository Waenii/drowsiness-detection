# Driver Drowsiness Detection System

A Flask-based web application that detects driver drowsiness and yawning using computer vision. It leverages facial landmarks to monitor eye and mouth activity in real-time and logs events into a SQLite database.

## Features

- **Real-time Drowsiness Detection** using OpenCV, dlib, and facial landmark analysis.
- **Yawning Detection** using mouth aspect ratio.
- **Alarm Alert** using pygame to play a warning sound.
- **User Authentication**: Registration and login system.
- **Dashboard** with event logging and visual analytics.
- **Mobile-Responsive Frontend**: Styled pages for login, register, and dashboard.
- **Database Logging**: Events (drowsiness/yawning) logged into a SQLite database.
- **Deployment Ready**: Supports deployment on platforms like PythonAnywhere or Railway.

## Tech Stack

- **Frontend**: HTML, CSS (Responsive & Stylish UI)
- **Backend**: Python, Flask
- **Computer Vision**: OpenCV, dlib, NumPy, SciPy
- **Database**: SQLite
- **Authentication**: Flask-Bcrypt for password hashing
- **Deployment**: PythonAnywhere / Railway

## Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/drowsiness-detection.git
cd drowsiness-detection
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

Access the app at `http://localhost:5000`.

## File Structure

```
- app.py
- requirements.txt
- runtime.txt
- users.db
- alarm.wav
- shape_predictor_68_face_landmarks.dat
- /templates
    - login.html
    - register.html
    - dashboard.html
- /static
    - /images
```

## Important Notes

- Ensure `shape_predictor_68_face_landmarks.dat` is in the root directory. You can download it from [dlib's model repository](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
- You must allow access to your camera for real-time detection to work.
- Drowsiness and yawning events will be saved automatically while the detection is running.

## Screenshots
Login Page
![LoginPage](https://github.com/user-attachments/assets/a2aeab1c-b852-4696-92e7-c47f18606fdb)

Dashboard
![dashboard](https://github.com/user-attachments/assets/83afc4a2-f667-441f-b468-83f56ae5ea14)


## License

This project is for academic and demonstration purposes only.
