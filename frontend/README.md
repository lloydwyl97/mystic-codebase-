# 🚫 WARNING: FRONTEND DASHBOARD IS DISABLED

This folder is blocked for dashboard development.

✅ Use ONLY:
`services/mystic_super_dashboard/app/main.py`

❌ DO NOT:
- Add new dashboard files here
- Reference this folder in any Dockerfile or Compose file
- Create streamlit or dash dashboards here

## 🚫 BLOCKED OPERATIONS:
- Adding `streamlit_dashboard.py`, `app.py`, or any dashboards here
- Referencing this folder in Dockerfiles or docker-compose.yml
- Creating new dashboard files in this directory
- Placing any Streamlit, Dash, or web dashboard code here

## ✅ CORRECT LOCATION:
All dashboard development must be done in:
- `services/mystic_super_dashboard/app/main.py` - Main dashboard application
- `services/mystic_super_dashboard/app/pages/` - Dashboard pages
- `services/mystic_super_dashboard/Dockerfile` - Dashboard container

## 🐳 DOCKER SERVICE:
The dashboard is defined in the main `docker-compose.yml` as:
```yaml
dashboard:
  build:
    context: ./services/mystic_super_dashboard
  container_name: mystic_super_dashboard
  ports:
    - "8501:8501"
```

**🚨 ANY DASHBOARD CODE PLACED IN THIS FRONTEND FOLDER WILL BE DELETED.**
