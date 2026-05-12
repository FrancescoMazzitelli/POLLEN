// GET list dispatcher for TMDB search endpoints
def query = request.headers['query'] ?: ''
def path = request.path

def results = [
    person: [
        results: [
            [id: 1769, name: 'Sofia Coppola', known_for_department: 'Directing'],
            [id: 18918, name: 'Francis Ford Coppola', known_for_department: 'Directing']
        ]
    ],
    movie: [
        results: [
            [id: 238, title: 'The Godfather', release_date: '1972-03-24'],
            [id: 680, title: 'Pulp Fiction', release_date: '1994-10-14']
        ]
    ]
]

if (path.contains('person')) return new JsonResponse(results.person)
if (path.contains('movie'))  return new JsonResponse(results.movie)
return new JsonResponse([results: []])
