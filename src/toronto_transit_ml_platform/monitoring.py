from prometheus_client import Counter

prediction_requests = Counter(
	"prediction_requests_total",
	"Total number of successful prediction requests",
)
