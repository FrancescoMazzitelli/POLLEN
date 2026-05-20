"""REST controller exposing the Control Unit orchestration endpoint."""

from flask import request, jsonify
from flask_restx import Namespace, Resource, reqparse
from service.controlService import Controller

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
        """Accept a user query and return an execution plan with results."""
        data = request.get_json(force=True)
        user_input = data["input"]
        print(f"Input received: {user_input}")

        controller = Controller()
        results = controller.control(user_input)
        return jsonify(results)
