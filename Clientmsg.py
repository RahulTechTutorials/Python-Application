import socket
mySock = socket.socket()
host = '192.168.0.102'
port = 65535
mySock.connect((host,port))
while True:
    data = input('Enter the data to be sent')
    if data == '':
        mySock.send('EOF'.encode())
        break
    else:
        mySock.send(data.encode())
    data = mySock.recv(1024)
    print(data.decode())
mySock.close()
