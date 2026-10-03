#!/bin/bash

echo "==================================================="
echo "Starting Rehabilitation Exercise Assessment Pipeline"
echo "==================================================="

# Step 1 & 2: Feature Extraction and Temporal Segmentation
echo "[1/2] Running Feature Extraction and Temporal Segmentation..."
echo "This will process all cam0.mp4 videos in the dataset."
python feature_extraction_opencv.py

if [ $? -ne 0 ]; then
    echo "Error during feature extraction. Pipeline aborted."
    exit 1
fi

echo ""
echo "[2/2] Running Quality Scoring Model..."
# Step 3: Train models on the extracted dataset
python scoring_model.py

echo ""
echo "==================================================="
echo "Pipeline Execution Completed!"
echo "==================================================="
