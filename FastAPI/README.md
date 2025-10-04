## Punters Toolkit Backend

First, open a virtual environment:

if you haven't created one yet
```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the API server:

```bash
uvicorn main:app --reload
```
