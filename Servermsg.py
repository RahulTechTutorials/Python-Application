import socket
mySock = socket.socket()
host = '192.168.0.102'
port = 65535
mySock.bind((host,port))
mySock.listen(1)
conn, addr = mySock.accept()
while True:
    data = conn.recv(1024)
    if data.decode() == 'EOF':
        break
        conn.send(data)
conn.close()
mySock.close()
