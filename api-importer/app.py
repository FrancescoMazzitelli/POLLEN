from flask import Flask
from flask_restx import Api
from controller import importController
from cheroot.wsgi import Server

app = Flask(__name__)
app.config['BUNDLE_ERRORS'] = True

api = Api(app,
          title="Api importer",
          version="1.0",
          description="API documentation for external services importer and data processing operations",
          doc="/swagger")

BASE_PATH = "/api"

api.add_namespace(importController.api,       path=f"{BASE_PATH}/importer")

if __name__ == "__main__":
    server = Server(("0.0.0.0", 7500), app)
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()