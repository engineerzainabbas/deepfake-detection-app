# 🚀 GitHub Upload Guide for Streamlit Deployment

## Step 1 — Create GitHub Repository
1. Go to github.com
2. Click "New Repository"
3. Name: deepfake-detection-app
4. Set to PUBLIC ✅ (required for free Streamlit Cloud)
5. Click "Create repository"

## Step 2 — Upload These Files
Upload ALL of these files to your repo root:

| File | Where to get it |
|---|---|
| deepfake_dashboard_final.py | Downloaded from Claude |
| best_deepfake.keras | In your notebook folder |
| model_info.json | In your notebook folder |
| requirements.txt | Downloaded from Claude |
| packages.txt | Downloaded from Claude |
| .streamlit/config.toml | Downloaded from Claude |

⚠️ IMPORTANT: best_deepfake.keras might be large (50-100MB)
GitHub allows max 100MB per file.
If it is over 100MB use Git LFS (instructions below).

## Step 3 — Check model file size
Run this in your notebook:
```python
import os
size_mb = os.path.getsize('best_deepfake.keras') / (1024*1024)
print(f'Model size: {size_mb:.1f} MB')
```

If size > 100MB → use Git LFS:
```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes
```

If size < 100MB → upload normally.

## Step 4 — Deploy on Streamlit Cloud
1. Go to share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your repo: deepfake-detection-app
5. Branch: main
6. Main file path: deepfake_dashboard_final.py
7. Click "Deploy!"

## Step 5 — Wait for deployment
- Takes 3-5 minutes first time
- Watch the logs for any errors
- Your app URL will be:
  https://your-username-deepfake-detection-app.streamlit.app

## Common Errors & Fixes

| Error | Fix |
|---|---|
| Module not found | Check requirements.txt spelling |
| Model not found | Make sure best_deepfake.keras is in repo root |
| Memory error | Switch to tensorflow-cpu in requirements.txt |
| File too large | Use Git LFS for .keras file |
