# Automated Attendance System using Facial Recognition

This project is an implementation of an automated attendance system using facial recognition technology. It detects and identifies faces in real-time using a webcam, matches them against pre-trained images, and records attendance based on the time a student is present.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [How It Works](#how-it-works)
- [Results and Accuracy](#results-and-accuracy)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The Automated Attendance System is designed to automate the process of tracking student attendance using facial recognition. It captures real-time video feed, recognizes faces, and records attendance in a CSV file. The project also calculates and displays recognition accuracy over time.

## Features
- **Real-time Face Detection and Recognition**: The system uses a webcam to detect and recognize faces.
- **Attendance Tracking**: Automatically records attendance when a recognized face is detected for a sufficient period of time.
- **Data Logging**: Logs attendance data into a CSV file with details such as NIM (Student ID), name, date, time, and status.
- **Accuracy Calculation**: Tracks and displays the accuracy of the recognition system over time.
- **Result Plotting**: Generates a plot to visualize recognition accuracy as a function of elapsed time.

## Prerequisites
Ensure you have the following packages installed in your Python environment:

- `numpy`
- `opencv-python`
- `face_recognition`
- `pandas`
- `matplotlib`

You can install the required packages using pip:

```bash
pip install numpy opencv-python face_recognition pandas matplotlib
```

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/automated-attendance-system.git
    cd automated-attendance-system
    ```

2. **Prepare training images**:
    - Place all images in the `training/` directory.
    - Images should be named in the format `NIM-Name.jpg` or `NIM_Name.jpg`.
    - Example: `123456-JohnDoe.jpg`.

3. **Run the script**:
    ```bash
    python attendance_system.py
    ```

## Usage

- Run the script `attendance_system.py` to start the attendance system.
- The system will access the webcam, detect faces, and start recognizing them against the training images.
- The recognized attendance will be saved in `Attendance.csv`.
- To stop the system, press the `q` key.

## File Structure

```
.
├── training/               # Directory containing training images
├── attendance_system.py    # Main script for running the attendance system
├── Attendance.csv          # CSV file where attendance records are saved (generated after running)
├── recognition_results.csv # CSV file containing recognition results over time (generated after running)
└── README.md               # This README file
```

## How It Works

1. **Face Encodings**: The script generates encodings for faces found in the images inside the `training/` directory.
2. **Real-time Recognition**: The script captures video from the webcam, detects faces, and compares them with the encoded faces.
3. **Attendance Logging**: When a recognized face is detected for at least 10% of the predefined attendance time, it is logged in the `Attendance.csv` file.
4. **Accuracy Calculation**: The script tracks correct and total recognitions to calculate and display accuracy over time.
5. **Plotting**: A plot is generated to visualize the accuracy of face recognition over time.

## Results and Accuracy

After running the script, you can find the recognition accuracy plotted against time in a graph. The `recognition_results.csv` file contains raw data used for the accuracy calculation.
