from flask import request, jsonify, Response
from flask_restx import Namespace, Resource, reqparse
from werkzeug.datastructures import FileStorage
from service.discoveryService import Discovery
from service.controlService import Controller
from langchain_ollama import ChatOllama

api = Namespace("control", description="Services management and orchestration")
control_unit_parser = reqparse.RequestParser()

category_model = api.schema_model(
    "CategorySchema",
    {
        "type": "object",
        "additionalProperties": {
            "type": "string",
        },
        "example": {
            "input": "your text here"
        },
    },
)

@api.route("/invoke")
@api.expect(category_model)
class ConversationalAgent(Resource):
    def post(self):

        data = request.get_json(force=True)
        user_input = data['input']
        print(f"Input ricevuto: {user_input}")

        controller = Controller()
        results = controller.control(user_input)
        return jsonify(results)

