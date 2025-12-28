# Production Deployment Guide

To bring the **Wind Turbine Fault Detector** to production, follow these steps. I have already prepared the necessary configuration files for you.

## 1. Choose a Platform
You have several options for hosting your Flask application:
- **Render (Recommended)**: Very easy to set up, supports Docker and automatic deployments from GitHub.
- **Heroku**: Industry standard for PaaS, handles Python environments well.
- **DigitalOcean App Platform**: Similar to Render, great performance.
- **AWS/GCP (Advanced)**: For more control and scalability.

## 2. Configuration Files (Already Created)
I have added the following files to your project:
- `requirements.txt`: Lists all Python libraries needed.
- `Procfile`: Tells the hosting service how to run your app using `gunicorn`.
- `runtime.txt`: Specifies the Python version.
- `Dockerfile`: Allows you to run the app in a container for maximum consistency.

## 3. Deployment Steps (Render/Heroku)
1. **Push to GitHub**: Make sure your code is on GitHub.
2. **Handle Large Files**: 
   - Your model files (`final_model.joblib`, `shap_explainer.joblib`) are large (~200MB total).
   - If using Git, it is highly recommended to use **Git LFS** (Large File Storage) to track these files.
3. **Set Up Web Service**:
   - On Render: Create a "Web Service", connect your repo, and it will automatically detect the settings.
   - On Heroku: `heroku create`, then `git push heroku main`.

## 4. Performance & Scalability
- **WSGI Server**: I've configured the app to use `gunicorn` in production. This handles multiple requests simultaneously, unlike the Flask development server.
- **Memory Limits**: Since you are loading large ML models into memory, ensure your hosting plan has at least **512MB - 1GB of RAM**.

## 5. Security
- **Debug Mode**: The code I updated in `final_app.py` automatically disables `debug` mode in production.
- **Secret Key**: If you add sessions or forms in the future, set a `SECRET_KEY` environment variable.

## 6. How to Run Locally in Production Mode
If you want to test how it will run in production:
```bash
# Using gunicorn (Unix-like systems)
gunicorn final_app:app

# Using Docker
docker build -t wind-turbine-detector .
docker run -p 5000:5000 wind-turbine-detector
```
