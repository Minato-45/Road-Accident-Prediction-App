#!/bin/bash
# Build script for Render deployment

echo "Python version check:"
python --version
python3 --version

echo "Installing dependencies with no-cache to ensure fresh install..."
pip install --no-cache-dir --upgrade pip

echo "Installing requirements with pre-compiled wheels only..."
pip install --no-cache-dir --only-binary=all -r requirements.txt

echo "Verifying installations..."
python -c "import flask; print(f'Flask: {flask.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
python -c "import numpy; print(f'Numpy: {numpy.__version__}')"

echo "Build completed successfully!"