# Gel Genotyper – ARMS PCR Gel Image Analyzer

This web app automatically analyzes gel electrophoresis images of ARMS PCR for two specific mutations (SCA and IVS1-5). It identifies band sizes based on a 100 bp ladder and infers genotypes using reference patterns.

---

## 🧬 Supported Band Sizes

- **Control**: 782 bp
- **SCA Wild**: 570 bp
- **SCA Mutant**: 266 bp
- **IVS1-5 Wild**: 344 bp
- **IVS1-5 Mutant**: 485 bp

---

## 🚀 Features

- Upload gel image
- Auto-detect lanes and bands
- Calibrate with user-specified 100 bp ladder lane
- Return band sizes and inferred genotype
- Results copyable for Excel export

---

## 🛠 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/gel-genotyper.git
cd gel-genotyper

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python gel_genotyper.py
