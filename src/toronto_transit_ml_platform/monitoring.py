from prometheus_client import Counter, Histogram

prediction_requests = Counter(
	"prediction_requests_total",
	"Total number of successful prediction requests",
)

prediction_latency = Histogram(
	"prediction_latency_seconds",
	"Time spent processing prediction requests",
)
