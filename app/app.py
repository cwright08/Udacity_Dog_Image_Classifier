from resources import app
from resources import model_prediction
from resources import routes

app.run(host='0.0.0.0', port=3001, debug=False)