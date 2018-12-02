# flask-ml-serve
Flask API to serve ModHero NLP ML model results - deployed to heroku

## Getting Started
installation 
1. Setup python virtual env
3. `pip install -r requirements.txt`

running & testing
4. `gunicorn src.serve:app` 
5. `curl -d "text=this is an example" -X POST http://localhost:8000/v1/api/classify`

## Built With
scikit-learn
flask
gunicorn