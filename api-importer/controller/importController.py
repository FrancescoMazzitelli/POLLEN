"""REST controller exposing API import and schema enrichment endpoints."""

from service import importService
from flask import request, jsonify
from flask_restx import Namespace, Resource, reqparse
from werkzeug.datastructures import FileStorage
import json
import os
import tempfile

api = Namespace("importer", description="External API import and management")
import_parser = reqparse.RequestParser()

upload_parser = reqparse.RequestParser()
upload_parser.add_argument("file",       location="files", type=FileStorage, required=True, help="OpenAPI YAML or JSON file")
upload_parser.add_argument("id",         location="form",  type=str,         required=True, help="Service ID")
upload_parser.add_argument("service_id", location="form",  type=str,         required=False)
upload_parser.add_argument("base_url", location="form", type=str, required=False, help="Fallback base URL if missing in OpenAPI")


@api.route("/import")
class ImportAPI(Resource):
    def post(self):
        """Import all providers from apis.guru."""
        import_service = importService.Service()
        try:
            import_service.import_apis()
        except Exception as e:
            return {"error": str(e)}, 500


@api.route("/import/<key>")
class ImportSpecificAPI(Resource):
    def post(self, key):
        """Import a single provider from apis.guru by key (e.g. stripe.com)."""
        import_service = importService.Service()
        try:
            providers = import_service.fetch_providers()
            if key not in providers:
                return {"message": f"No provider found for key: {key}"}, 404
            api_data    = import_service.fetch_api_details(key)
            swagger_url = import_service.extract_swagger_url(api_data, key)
            openapi     = import_service.parse_swagger(key, swagger_url)
            if not openapi:
                return {"error": f"Failed to parse swagger for {key}"}, 500
            import_service.register_to_redis(key, "true")
            import_service.register_to_consul(key, openapi["name"])
            import_service.register_to_mongo(openapi)
            return {"status": "ok", "id": key, "name": openapi.get("name")}, 200
        except Exception as e:
            return {"error": str(e)}, 500


@api.route("/import/url")
class ImportFromUrl(Resource):
    def post(self):
        """Import a service from a public OpenAPI YAML/JSON URL.

        JSON body:
          {
            "id":          "restcountries",          (required - unique service ID)
            "swagger_url": "https://..."             (required - URL to OpenAPI spec)
          }
        """
        data = request.get_json(force=True)
        if not data:
            return {"error": "Missing JSON body"}, 400

        service_id  = data.get("id")
        swagger_url = data.get("swagger_url")

        if not service_id:
            return {"error": "Missing required field: 'id'"}, 400
        if not swagger_url:
            return {"error": "Missing required field: 'swagger_url'"}, 400

        import_service = importService.Service()
        try:
            openapi = import_service.parse_swagger(service_id, swagger_url)
            if not openapi:
                return {"error": f"Failed to parse OpenAPI spec from {swagger_url}"}, 500

            import_service.register_to_redis(service_id, "true")
            import_service.register_to_consul(service_id, openapi["name"])
            import_service.register_to_mongo(openapi)

            return {
                "status":    "ok",
                "id":        service_id,
                "name":      openapi.get("name"),
                "endpoints": len(openapi.get("endpoints", {})),
            }, 200

        except Exception as e:
            return {"error": str(e)}, 500


@api.route("/import/file")
class ImportFromFile(Resource):
    @api.expect(upload_parser)
    def post(self):
        """Import a service by uploading an OpenAPI YAML or JSON file directly.

        Form data:
          file - the OpenAPI .yaml / .json file (required)
          id   - unique service ID           (required)
        """
        args = upload_parser.parse_args()
        uploaded_file = args["file"]
        service_id    = args.get("id") or args.get("service_id")
        base_url      = args.get("base_url")

        if not service_id:
            return {"error": "Missing required field: 'id'"}, 400
        if not uploaded_file:
            return {"error": "Missing required field: 'file'"}, 400

        suffix = os.path.splitext(uploaded_file.filename)[1] or ".yaml"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                uploaded_file.save(tmp)
                tmp_path = tmp.name
        except Exception as e:
            return {"error": f"Failed to save uploaded file: {e}"}, 500

        try:
            swagger_url = f"file://{tmp_path}"

            import_service = importService.Service()
            openapi = import_service.parse_swagger(service_id, swagger_url, fallback_base_url=base_url)
            if not openapi:
                return {"error": "Failed to parse uploaded OpenAPI spec"}, 500

            import_service.register_to_redis(service_id, "true")
            import_service.register_to_consul(service_id, openapi["name"])
            import_service.register_to_mongo(openapi)

            return {
                "status":    "ok",
                "id":        service_id,
                "name":      openapi.get("name"),
                "endpoints": len(openapi.get("endpoints", {})),
            }, 200

        except Exception as e:
            return {"error": str(e)}, 500

        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@api.route("/enrich")
class EnrichSchemas(Resource):
    def post(self):
        """Download YAML specs from Microcks via prance, extract response_schemas
        for each registered service in MongoDB, and update documents via
        PATCH /services/<id>/schemas on the catalog-gateway.

        Optional JSON body:
          { "mock_server_url": "http://mock-server:8080" }
          { "service_id": "smart-charging-stations-mock" }

        Returns:
          { "enriched": 5, "skipped": 1, "errors": 0 }
        """
        data            = request.get_json(silent=True) or {}
        mock_server_url = data.get("mock_server_url") or os.environ.get("MOCK_SERVER_URL")
        service_id      = data.get("service_id")

        import_service = importService.Service()
        try:
            result = import_service.enrich_schemas(
                mock_server_url=mock_server_url,
                service_id=service_id
            )
            return result, 200
        except Exception as e:
            return {"error": str(e)}, 500
