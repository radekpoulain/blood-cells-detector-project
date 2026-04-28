# 🚀 Deployment Guide: Streamlit Community Cloud

Follow these steps to deploy your **Blood Cell Detector** to the web for free.

## 1. Prepare your GitHub Repository
Ensure you have forked the repository to your account and pushed the latest changes.
The repository must contain:
- `app.py` (The main application)
- `requirements.txt` (The list of dependencies)
- `blood_detector_model.pt` (The trained weights)
- `test_images/` (Folder with sample images)

## 2. Sign in to Streamlit
1. Go to [share.streamlit.io](https://share.streamlit.io/).
2. Sign in with your GitHub account.

## 3. Deploy the App
1. Click the **"New app"** button.
2. Select your forked repository: `your-username/blood-cells-detector-project`.
3. Set the **Main file path** to `app.py`.
4. (Optional) Customize the **App URL** to something like `blood-cell-detector`.
5. Click **"Deploy!"**.

## 4. Resource Configuration
Streamlit Cloud provides sufficient resources for this model (CPU-only).
- **RAM**: Up to 1GB (The model uses ~200MB during inference).
- **Disk**: The model is ~44MB.

## 5. Troubleshooting
- **ModuleNotFoundError**: Ensure `requirements.txt` is in the root directory.
- **Model not found**: Ensure `blood_detector_model.pt` is in the root directory.
- **Memory errors**: If the app crashes on large images, try reducing `max_det` in the sidebar.

---
*Created by Antigravity AI*
