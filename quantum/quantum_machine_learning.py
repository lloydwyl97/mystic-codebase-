import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
import pennylane as qml
import numpy as np
import os

app = FastAPI(
    title="Mystic PennyLane Quantum Service",
    description="Quantum machine learning engine powered by PennyLane.",
    version="1.0.0",
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "pennylane_requests_total", "Total PennyLane API Requests", ["endpoint"]
)


@app.get("/health", tags=["Health"])
def health():
    REQUEST_COUNT.labels(endpoint="/health").inc()
    return {"status": "ok", "service": "pennylane", "version": "1.0.0"}


@app.get("/metrics")
def metrics():
    return JSONResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/quantum/ml", tags=["Quantum"])
def quantum_ml(num_qubits: int = 2, layers: int = 2):
    REQUEST_COUNT.labels(endpoint="/quantum/ml").inc()
    try:
        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev)
        def quantum_circuit(weights):
            for i in range(num_qubits):
                qml.Hadamard(wires=i)
            for layer in range(layers):
                for i in range(num_qubits):
                    qml.Rot(*weights[layer, i], wires=i)
                qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        weights = np.random.random((layers, num_qubits, 3))
        result = quantum_circuit(weights)
        return {
            "result": float(result),
            "num_qubits": num_qubits,
            "layers": layers,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8089))
    uvicorn.run(app, host="0.0.0.0", port=port)
