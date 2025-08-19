import os

import cirq
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

app = FastAPI(
    title="Mystic Cirq Quantum Service",
    description="Quantum optimization engine powered by Cirq.",
    version="1.0.0",
)

# Prometheus metrics
REQUEST_COUNT = Counter("cirq_requests_total", "Total Cirq API Requests", ["endpoint"])


@app.get("/health", tags=["Health"])
def health():
    REQUEST_COUNT.labels(endpoint="/health").inc()
    return {"status": "ok", "service": "cirq", "version": "1.0.0"}


@app.get("/metrics")
def metrics():
    return JSONResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/quantum/optimize", tags=["Quantum"])
def quantum_optimize(num_qubits: int = 2, depth: int = 4):
    REQUEST_COUNT.labels(endpoint="/quantum/optimize").inc()
    try:
        qubits = cirq.LineQubit.range(num_qubits)
        circuit = cirq.Circuit()
        for i in range(depth):
            circuit.append(cirq.H.on_each(qubits))
            circuit.append(cirq.CNOT(qubits[0], qubits[1]))
        simulator = cirq.Simulator()
        simulator.run(circuit, repetitions=1000)
        return {
            "circuit": str(circuit),
            "num_qubits": num_qubits,
            "depth": depth,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8088))
    uvicorn.run(app, host="0.0.0.0", port=port)
