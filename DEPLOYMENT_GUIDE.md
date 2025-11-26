# Deployment Guide for Render

## Prerequisites
1. GitHub account
2. Render account (free at render.com)
3. Git installed on your computer

## Step-by-Step Deployment Instructions

### 1. Prepare Your Code
✅ **Already Done:** Your project has been configured with:
- Updated `requirements.txt` with compatible versions
- `Procfile` for Render deployment
- `runtime.txt` specifying Python version
- `render.yaml` for Render configuration
- Updated static file paths in templates
- Configured Flask for production hosting

### 2. Create a GitHub Repository
1. Go to https://github.com and create a new repository
2. Name it something like "road-accident-prediction"
3. Make it public (required for free Render hosting)
4. Don't initialize with README (your project already has one)

### 3. Upload Your Code to GitHub
Open a terminal in your project directory and run:
```bash
git init
git add .
git commit -m "Initial commit - Road Accident Prediction App"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/road-accident-prediction.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

### 4. Deploy on Render
1. Go to https://render.com and sign up/log in
2. Click "New +" and select "Web Service"
3. Choose "Build and deploy from a Git repository"
4. Connect your GitHub account if not already connected
5. Select your repository "road-accident-prediction"
6. Configure the deployment:
   - **Name**: road-accident-prediction (or any name you prefer)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free
7. Click "Create Web Service"

### 5. Wait for Deployment
- Render will automatically detect your configuration files
- The deployment process takes about 5-10 minutes
- You can watch the build logs in real-time

### 6. Access Your Application
Once deployment is complete, Render will provide you with a URL like:
`https://road-accident-prediction-XXXX.onrender.com`

## Important Notes

### File Structure Maintained
Your app will work exactly like the localhost version because:
- All static file paths have been converted to use Flask's `url_for()`
- The machine learning model (`model.pkl`) will be included
- All routes and functionality remain the same

### Features Available
✅ Home page with prediction form
✅ Data upload and preview functionality  
✅ Login page
✅ Performance charts
✅ All static assets (CSS, JavaScript, images)
✅ Machine learning predictions

### Troubleshooting
If the deployment fails:
1. Check the build logs on Render dashboard
2. Ensure all files are committed to GitHub
3. Verify `requirements.txt` has all dependencies
4. Check that `model.pkl` file is included in the repository

### Free Tier Limitations
- App may sleep after 15 minutes of inactivity
- Takes 30-60 seconds to wake up when accessed after sleeping
- 750 build hours per month (should be sufficient for most use cases)

### Optional: Custom Domain
After successful deployment, you can:
1. Purchase a domain name
2. Configure it in Render dashboard
3. Render provides free SSL certificates

Your Road Accident Prediction app will be fully functional on Render and accessible worldwide!