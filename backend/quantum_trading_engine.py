import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from qiskit import QuantumCircuit, execute, Aer
import os

app = FastAPI(
    title="Mystic Qiskit Quantum Service",
    description="Quantum trading engine powered by Qiskit.",
    version="1.0.0",
)

# Prometheus metrics
REQUEST_COUNT = Counter("qiskit_requests_total", "Total Qiskit API Requests", ["endpoint"])


@app.get("/health", tags=["Health"])
def health():
    REQUEST_COUNT.labels(endpoint="/health").inc()
    return {"status": "ok", "service": "qiskit", "version": "1.0.0"}


@app.get("/metrics")
def metrics():
    return JSONResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/quantum/execute", tags=["Quantum"])
def quantum_execute(num_qubits: int = 2, shots: int = 1024):
    REQUEST_COUNT.labels(endpoint="/quantum/execute").inc()
    try:
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.cx(0, 1)
        backend = Aer.get_backend("qasm_simulator")
        job = execute(qc, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        return {"result": counts, "num_qubits": num_qubits, "shots": shots}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8087))
    uvicorn.run(app, host="0.0.0.0", port=port)
