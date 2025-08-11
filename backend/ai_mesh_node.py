import socket


def start_node(host="localhost", port=9444):
    s = socket.socket()
    s.bind((host, port))
    s.listen(5)
    print(f"[MESH] Listening on {host}:{port}")
    while True:
        conn, addr = s.accept()
        data = conn.recv(1024).decode()
        print(f"[MESH] Received from {addr}: {data}")
        conn.close()
