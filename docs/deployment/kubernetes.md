# Kubernetes Deployment

Manifests live under `k8s/`. They target standard nginx-ingress + cert-manager.

## Apply

```bash
kubectl apply -f k8s/namespace.yaml
kubectl create secret generic ctxai-secrets \
  -n ctxai \
  --from-literal=openrouter-api-key="$OPENROUTER_API_KEY"
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/ingress.yaml
```

## Components

- `namespace.yaml` — creates the `ctxai` namespace.
- `deployment.yaml` — 2-replica deployment with readiness/liveness probes,
  resource requests, and `prometheus.io/scrape` annotations.
- `service.yaml` — ClusterIP service + a PVC for `/data`.
- `ingress.yaml` — TLS-terminated nginx ingress.

## Scaling

```bash
kubectl scale deployment/ctxai -n ctxai --replicas=5
```

For autoscaling, install metrics-server and apply an HPA:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ctxai
  namespace: ctxai
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ctxai
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

## Monitoring

The deployment exposes `/api/v1/metrics` in Prometheus exposition format.
Pair with a `ServiceMonitor` (prometheus-operator) or static scrape target.
