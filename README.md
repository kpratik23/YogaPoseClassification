# Real-Time Yoga Pose Identification

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/your-username/your-repo)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Description

This project utilizes computer vision techniques to identify yoga poses in real time using a camera. It supports the following four poses:

- Tree Pose
- Warrior Pose
- Downward Dog Pose
- No Pose Detected

The system runs in real time at up to 30 frames per second (fps), providing instant feedback on the user's pose.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Demo](#demo)
- [Credits](#credits)
- [License](#license)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/kpratik23/YogaPoseClassification.git
2. Navigate to the project directory:
   
   ```bash
   cd YogaPoseClassification
3. Create a virtual environment (optional but recommended):
   - On Windows
     
     ```bash
     venv\Scripts\activate
     
    - On macOS/Linux
      
      ```bash
      source venv/bin/activate
4. Install the required dependencies:
  
    ```bash
    pip install -r requirements.txt
    
## Usage
The application supports different modes for pose identification:

- Camera Mode
  To use your camera for real-time pose identification, run the following command:

  ```bash
  python main.py --camera

- Image Mode
  If you want to identify a pose from an image file, use the following command:

  ```bash
  python main.py --image /path/to/your/image.jpg
- Video Mode
  For pose identification from a video file, use the following command:

  ```bash
  python main.py --video /path/to/your/video.mp4

## Features
  1. Real-time identification of yoga poses.
  2. Support for three common yoga poses.
  3. High-speed processing at up to 30 frames per second.
  4. User-friendly interface for easy interaction.
     
## Demo


https://github.com/kpratik23/YogaPoseClassification/assets/141538334/db09029c-6082-48de-ba63-df90369ed129

https://github.com/kpratik23/YogaPoseClassification/assets/141538334/68570705-edb9-45fa-9eb2-d556d6e792d4

https://github.com/kpratik23/YogaPoseClassification/assets/141538334/fe42b282-9c6b-4a0a-9f7b-477657159398



## Credits

**Credit:** The videos for testing purpose are taken from - **Alo Moves - Online Yoga & Fitness Videos**
 - https://www.youtube.com/watch?v=Mn6RSIRCV3w
 - https://www.youtube.com/watch?v=EC7RGJ975iM&t=7s
 - https://www.youtube.com/watch?v=Fr5kiIygm0c




## License
  This project is licensed under the MIT License.


