// GET list dispatcher for Spotify /search
// Returns a selection of example responses based on the query parameter.
def q = request.headers['q'] ?: ''
def type = request.headers['type'] ?: 'track'

def results = [
    track: [
        items: [
            [id: '1xBnJqBbHcZx8l7P3kLmN', name: 'Mariah Carey - All I Want for Christmas Is You',
             artists: [[id: '1xBnJqBbHcZx8l7P3kLmN', name: 'Mariah Carey']]],
            [id: '2yCoPkRqTl9mN8xJ4kHpQ', name: 'Mariah Carey - We Belong Together',
             artists: [[id: '1xBnJqBbHcZx8l7P3kLmN', name: 'Mariah Carey']]]
        ]
    ]
]
return new JsonResponse(results.get(type, results.track))
