import requests
import json

class Discovery:
    _instance = None

    """
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Discovery, cls).__new__(cls)
        return cls._instance
    """

    def __init__(self, address):
        """
        Inizializza la classe con i dati di connessione.
        """
        self.registry_address = address

    def services(self):
        response = requests.get(f"{self.registry_address}/v1/agent/services")
        services_data = response.json()

        services_list = []
        for service_id, service_info in services_data.items():
            meta = service_info.get('Meta', {})
            catalog_id = meta.get('service_doc_id', {})

            service = {
                "id": service_info['ID'],
                "service": service_info['Service'],
                "catalog_id": catalog_id,
            }
            services_list.append(service)

        return services_list
