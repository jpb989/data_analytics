# Dashboard Module: `src/dashboard/`

This directory implements the interactive Streamlit monitoring command center and real-time perturbation sandbox.

For the exhaustive technical specification, UI layout, and caching mechanics, see the complete guide:
👉 [**Comprehensive Dashboard Documentation**](../../docs/folder_readmes/dashboard.md)

---

### Key Files in this Folder

| File | Class / Functions | Description |
| :--- | :--- | :--- |
| [`app.py`](app.py) | Streamlit Web Application | Features scenario replay, custom mutation sandbox, 4-tier triage status banners, Plotly distribution overlays, and empirical research findings. |

---

### How to Run

```powershell
# Launch the interactive Streamlit dashboard
.venv\Scripts\streamlit run src/dashboard/app.py
```

Open your browser at **http://localhost:8501**.
