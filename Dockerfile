# Use an official Python image with Debian (for building dlib)
FROM python:3.9-slim

# Install dependencies for dlib and OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libboost-thread-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy your project files into the image
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Start the Flask app
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
