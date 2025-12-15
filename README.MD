# Multi-Color Object Tracking

Real-time detection of the largest object of specific colors (Blue, Red, Green) using OpenCV and NumPy.  
Displays the percentage contribution of the largest object relative to all detected pixels of that color.

Full disclosure: I've never written a README before so I used ChatGPT's help.


## Requirements

- Python 3.12.10
- OpenCV  
- NumPy  

## Installation

1. Clone the repository:

```bash
git clone https://github.com/aatri/color-tracking.git
cd color-tracking
```

2. Create a virtual environment:
```bash
python -m venv .venv
# Activate the environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
## Usage

Run the program:

    python main.py

Press q to quit the program.

Notes

    1. Ensure your webcam is connected and has proper lighting.
    2. HSV color ranges can be adjusted in main.py for better detection accuracy.
    