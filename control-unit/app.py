from flask import Flask
from flask_restx import Api
from controller import controlUnitController
from cheroot.wsgi import Server

app = Flask(__name__)
app.config['BUNDLE_ERRORS'] = True

api = Api(app, 
          title="Control unit", 
          version="1.0", 
          description="API documentation for the control unit service",
          doc="/swagger")

BASE_PATH = "/api"

api.add_namespace(controlUnitController.api, path=f"{BASE_PATH}/control")

if __name__ == "__main__":
    server = Server(("0.0.0.0", 5500), app)
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()
