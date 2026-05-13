import socket, json
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 5005))
sock.settimeout(5.0)
print("Bekleniyor...")
data, _ = sock.recvfrom(65535)
print("VERİ GELDİ:", list(json.loads(data.decode()).keys()))
sock.close()