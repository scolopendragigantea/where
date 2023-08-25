import socket

def receive_wav_file(server_ip, server_port, save_path):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((server_ip, server_port))
        
        with open(save_path, 'wb') as file:
            while True:
                data = client_socket.recv(1024)
                if not data:
                    break
                file.write(data)

        print("파일 수신 완료")
    except Exception as e:
        print("파일 수신 중 오류 발생:", e)
    finally:
        client_socket.close()

def main():
    server_ip = '192.168.0.7'
    server_port = 1234  # 서버와 동일한 포트 번호를 사용해야 합니다.
    a = 0
    while 1:
        a=a+1
        save_path = str(a)+'.wav'
        receive_wav_file(server_ip, server_port, save_path)

if __name__ == '__main__':
    main()
