"""Interfaces with HashiCorp Consul for service registry discovery."""

import requests


class Discovery:
    """Singleton-style client for Consul agent API."""

    _instance = None

    def __init__(self, address: str):
        self.registry_address = address

    def services(self) -> list[dict]:
        """Retrieve all registered services from Consul, returning id, name, and catalog_id."""
        response = requests.get(f"{self.registry_address}/v1/agent/services")
        services_data = response.json()

        services_list = []
        for service_id, service_info in services_data.items():
            meta = service_info.get("Meta", {})
            catalog_id = meta.get("service_doc_id", {})

            service = {
                "id": service_info["ID"],
                "service": service_info["Service"],
                "catalog_id": catalog_id,
            }
            services_list.append(service)

        return services_list
