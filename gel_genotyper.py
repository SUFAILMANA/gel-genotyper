
# gel_genotyper.py
from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

BAND_SIZES = {
    'Control': 782,
    'SCA_Wild': 570,
    'IVS_Mutant': 485,
    'IVS_Wild': 344,
    'SCA_Mutant': 266
}

TOLERANCE = 10

def interpret_bands(bands):
    labels = []
    for size in bands:
        for label, ref in BAND_SIZES.items():
            if abs(size - ref) <= TOLERANCE:
                labels.append(label)
    if 'SCA_Wild' in labels and 'IVS_Wild' in labels:
        return 'SCA Wild / IVS Wild'
    elif 'SCA_Wild' in labels and 'IVS_Mutant' in labels:
        return 'SCA Wild / IVS Het'
    elif 'SCA_Mutant' in labels and 'IVS_Mutant' in labels:
        return 'Compound Het'
    elif 'SCA_Mutant' in labels and 'IVS_Wild' in labels:
        return 'SCA Mut / IVS Wild'
    elif 'SCA_Wild' in labels:
        return 'SCA Het / IVS Wild'
    elif 'IVS_Mutant' in labels:
        return 'SCA Wild / IVS Mut'
    else:
        return 'Unclassified'

def detect_lanes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    vertical_proj = np.sum(thresh, axis=0)
    lane_positions = np.where(vertical_proj > np.max(vertical_proj) * 0.5)[0]
    lanes = []
    if len(lane_positions) > 0:
        start = lane_positions[0]
        for i in range(1, len(lane_positions)):
            if lane_positions[i] != lane_positions[i - 1] + 1:
                end = lane_positions[i - 1]
                lanes.append((start, end))
                start = lane_positions[i]
        lanes.append((start, lane_positions[-1]))
    return lanes

def detect_bands_in_lane(image, lane):
    lane_img = image[:, lane[0]:lane[1]]
    gray = cv2.cvtColor(lane_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal_proj = np.sum(thresh, axis=1)
    bands = np.where(horizontal_proj > np.max(horizontal_proj) * 0.5)[0]
    band_positions = []
    if len(bands) > 0:
        start = bands[0]
        for i in range(1, len(bands)):
            if bands[i] != bands[i - 1] + 1:
                end = bands[i - 1]
                band_positions.append((start + end) // 2)
                start = bands[i]
        band_positions.append((start + bands[-1]) // 2)
    return band_positions

def calibrate_ladder(ladder_band_positions):
    ladder_sizes = np.array([100 * i for i in range(1, len(ladder_band_positions) + 1)])
    distances = np.array(ladder_band_positions)
    log_sizes = np.log10(ladder_sizes)
    coeffs = np.polyfit(distances, log_sizes, 1)
    return coeffs

def estimate_band_size(y, coeffs):
    log_bp = coeffs[0] * y + coeffs[1]
    return round(10 ** log_bp)

def analyze_gel(image_path, ladder_lane_index):
    image = cv2.imread(image_path)
    lanes = detect_lanes(image)
    results = []
    if ladder_lane_index >= len(lanes):
        return [{"error": "Invalid ladder lane index."}]
    ladder_bands = detect_bands_in_lane(image, lanes[ladder_lane_index])
    coeffs = calibrate_ladder(ladder_bands)
    for i, lane in enumerate(lanes):
        if i == ladder_lane_index:
            continue
        band_positions = detect_bands_in_lane(image, lane)
        estimated_sizes = [estimate_band_size(y, coeffs) for y in band_positions]
        genotype = interpret_bands(estimated_sizes)
        results.append({
            "lane": f"Lane {i+1}",
            "bands": ", ".join(map(str, estimated_sizes)),
            "genotype": genotype
        })
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['gel_image']
        ladder_lane = int(request.form['ladder_lane']) - 1
        if file:
            filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            results = analyze_gel(filepath, ladder_lane)
            return render_template('results.html', results=results, image_filename=filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
