# My ML Project

This project is a full-stack machine learning application with the following components:

- **Backend**: FastAPI service that loads a trained PyTorch model and serves predictions.
- **Frontend (React)**: Single Page Application built with Vite/React, consuming the backend API.
- **Frontend (Streamlit Admin)**: Admin dashboard for model monitoring and testing.
- **MLflow**: Experiment tracking and artifact management.

---

## Project Structure

```
my_ml_project/
│
├── .github/
│   └── workflows/
│       └── ci.yml         # GitHub Actions
├── backend/
│   ├── app/               # Python package
│   │   ├── checkpoints/
│   │   │   └── model.pt
│   │   ├── tests/         # Unit tests
│   │   │   └── test_math.py
│   │   │   └── test_train.py
│   │   ├── model.py       # FeedForwardNet
│   │   ├── train.py       # loading & serving logic
│   │   ├── test.py        # loading & serving logic
│   │   └── api.py         # FastAPI routes
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── requirements-dev.txt
│
├── frontend/
│   ├── node‑app/           # React/Node.js SPA
│   │   ├── src/
│   │   │   ├── App.jsx
│   │   │   └── main.jsx
│   │   ├── Dockerfile
│   │   ├── index.html
│   │   ├── package.json
│   │   └── vite.config.js
│   └── streamlit‑admin/    # Admin dashboard
│       ├── app.py
│       ├── Dockerfile
│       └── requirements.txt
│
├── .gitignore
└── docker‑compose.yml
```

---

## Prerequisites

- Install **Docker** and **Docker Compose** (v2+).
- Ensure ports `8000`, `5173`, `8501`, and `5000` are free.

---

## Running the Project

### 1. Build and Start Services

```bash
docker compose up --build
```
or (for detached mode)
```bash
docker compose up --build -d
```

Then, use
```bash
docker compose up
```
to spin up the following services:

- **Backend**: FastAPI at [http://localhost:8000](http://localhost:8000)
- **Frontend (React)**: SPA at [http://localhost:5173](http://localhost:5173)
- **Frontend (Streamlit Admin)**: Admin dashboard at [http://localhost:8501](http://localhost:8501)
- **MLflow**: Tracking server at [http://localhost:5000](http://localhost:5000)


### 2. Stop Services

```bash
docker compose down
```

If you also want to remove volumes (MLflow DB, cached data):

```bash
docker compose down -v
```

---

## Running Training Jobs

Run the training script inside the backend container:

```bash
docker compose run --rm backend python app/train.py
```
or (for more parameters)
```bash
docker compose run --rm backend python -m app.train --epochs 20 --lr 0.01 --batch-size 32 --device cpu
```

Trained models will be saved in `backend/app/checkpoints/`.

---

## Running Tests

Run tests using pytest inside the backend container:

```bash
docker compose run --rm backend pytest app/tests
```

Tests also run automatically on GitHub Actions for every pull request targeting `main`.

---

## Running Evaluation (test.py)

If you want to test model inference locally (not via FastAPI API routes):

```bash
docker compose run --rm backend python app/test.py
```

---

## Using MLflow

Pull MlFlow and run the container with

```bash
docker compose pull mlflow
docker compose up mlflow
```

The MLflow tracking server runs automatically as part of Docker Compose at:

[http://localhost:5000](http://localhost:5000)

By default, the backend logs training runs to MLflow (configured with:

```
MLFLOW_TRACKING_URI=http://mlflow:5000
```

in `docker-compose.yml`).

You can view experiments, runs, and artifacts in the MLflow UI.

---

## Environment Variables

Each service has its own environment variables, configured in `docker-compose.yml`:

- **Backend**
  - `PYTHONPATH=/app` (ensures imports work)
  - `MLFLOW_TRACKING_URI=http://mlflow:5000`
- **Frontend (React)**
  - `VITE_API_URL=http://backend:8000`
- **Frontend (Streamlit Admin)**
  - `BACKEND_URL=http://backend:8000`

---

## Logs

Check logs in real time with:

```bash
docker compose logs -f
```

---

## Future Improvements

- Add linting/formatting checks in CI (`flake8`, `black`).
- Add healthchecks for backend/frontend.
- Automate migrations for MLflow database.

---

## License

MIT