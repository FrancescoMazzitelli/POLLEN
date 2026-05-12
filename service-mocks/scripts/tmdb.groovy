// POST /register dispatcher for TMDB
def response = [
    status: "registered",
    service: "TMDB",
    message: "Mock service registered via Microcks"
]
return new JsonResponse(response)
