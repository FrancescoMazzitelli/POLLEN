// POST /register dispatcher for Spotify
// When Microcks receives a POST /register on this service, this script
// defines the dynamic response. The actual registration (Consul + MongoDB)
// is now handled by deploy.sh via the api-importer, so this is a simple
// success response.
def response = [
    status: "registered",
    service: "Spotify",
    message: "Mock service registered via Microcks"
]
return new JsonResponse(response)
