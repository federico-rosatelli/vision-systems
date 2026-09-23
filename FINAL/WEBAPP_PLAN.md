# Streamlit WebApp Implementation Plan

This document outlines the architecture, strategy, and task breakdown for building a Streamlit web application that showcases all experiments conducted in the Flea Beetle Damage Assessment project, and provides live inference capabilities for uploaded images.

## 1. Implementation Strategy

The web application will be built using **Streamlit**, providing a fast and interactive UI directly linked to our Python backend. 

### Core Architecture:
1. **Historical Experiments Dashboard (Data Viewer):** 
   - A dedicated section of the app will read directly from the `outputs/tables/` and `outputs/runs/` directories.
   - It will render DataFrames (using `st.dataframe`) and historical plots to show the results of the MIL pipelines, Whole-Image baselines, and OOD zero-shot evaluations.
2. **Live Inference Engine (Model Server):**
   - We will use Streamlit's `@st.cache_resource` to load the heavy PyTorch models only once into memory.
   - When a user uploads an image, the app will run the image through the entire multi-stage pipeline:
     - **Classical CV**: Showing intermediate outputs like green-region plant proposals, thresholding masks, and hole-detection heuristics.
     - **MIL Pipeline**: Showing the bag-level prediction and rendering the learned Attention Map directly on the image to see where the model focuses.
     - **Detection (RFDetr)**: Rendering bounding boxes for direct hole pitting on the image.

### UI Layout:
- **Sidebar**: Navigation between "Project Overview", "Historical Results", "Live Inference", and "Learnings & Future Outlook". Model selection dropdown must allow the user to select from ALL trained models: `baseline_regression_seed42`, `baseline_regression_mse_seed42`, `patch_regression_seed42`, `mil_weighted_seed42`, `mil_abmil_seed42`, `patch_joint_seed42`, `patch_joint_sampled_seed42`, and `rfdetr_hole_pitting`.
- **Main Area**: Upload widgets, side-by-side image comparisons (Original vs Classical Segmentation vs MIL Attention Map), and markdown explanations.

---

## 2. Tasks

Use the standard tagging system: `[ ]` (Not Started), `[IN PROGRESS]` (Working on it), `[NEEDS VERIFICATION]` (Needs review), `[x]` (Completed).

- [x] **1. Project Learnings & Future Outlook Page**
  - [x] **Failed Approaches & Limitations:** Create a dedicated UI section detailing the limitations and failures across ALL pipelines:
    - *Classical CV:* Why heuristics (plant regions, thresholds) were fragile or failed.
    - *Whole-Image Baselines:* Why standard deep learning regression failed to capture localized damage.
    - *MIL Pipeline:* Specific edge cases or OOD environments where attention struggled.
    - *RFDetr Detection:* Limitations encountered during direct hole pitting annotations and detection.
  - [x] **Data Recording Recommendations:** Add a UI block outlining actionable tips and recommendations for recording video/image data in the field to make learning, bounding box annotations, and OOD generalization more feasible across all models.
  - [x] **Method Improvements:** Display an outlook on how all approaches (Classical CV pipelines, MIL aggregation, and RFDetr detection) could be improved further in future iterations.

- [x] **2. Project Initialization**
  - [x] Create the main entry point file: `scripts/app.py` (or `app/main.py`).
  - [x] Add `streamlit` and any missing UI dependencies to `requirements.txt`.
  - [x] Scaffold the basic sidebar navigation layout.

- [x] **3. Historical Results Dashboard**
  - [x] Build a page to load and visualize the OOD Evaluation Results (`ood_evaluation_results.csv`).
  - [x] Load and display the resistance leaderboards (`plot_resistance_leaderboard.csv`, `genotype_resistance_subset.csv`).
  - [x] Provide a UI to view the existing GT vs Prediction images from `outputs/rfdetr_hole_pitting/example_figures/`.

- [x] **4. Live Inference - Classical CV Integration**
  - [x] Implement a file uploader widget for `.jpg` / `.png` images.
  - [x] Import the classical CV heuristics (from `src/`).
  - [x] Add logic to process the uploaded image through the plant proposal pipeline and display the resulting segmentation masks side-by-side with the original image.

- [x] **5. Live Inference - Deep Learning Models (Baselines, MIL, RFDetr)**
  - [x] **Model Loading**: Implement an `@st.cache_resource` function to dynamically load ANY of the trained models chosen by the user (`baseline_regression_seed42`, `baseline_regression_mse_seed42`, `patch_regression_seed42`, `mil_weighted_seed42`, `mil_abmil_seed42`, `patch_joint_seed42`, `patch_joint_sampled_seed42`, `rfdetr_hole_pitting`).
  - [x] **Whole-Image Baselines**: Implement the standard forward pass for the whole-image models.
  - [x] **Patch & MIL Models**: Implement the patch extraction step, run the forward pass, and extract the attention weights/patch scores. Overlay them as a heatmap on the original image displaying it via `st.image()`.
  - [x] **RFDetr**: Implement the object detection forward pass. Draw the predicted bounding boxes on the image and display the count/confidence scores to the user.

- [x] **6. UI Polish and Testing**
  - [x] Ensure the app handles errors gracefully (e.g., if a user uploads a non-image file).
  - [x] Test memory consumption (ensure models aren't reloaded on every button click).
  - [x] Add explanatory markdown text throughout the app so external users understand the difference between the Classical CV and Deep Learning approaches.
