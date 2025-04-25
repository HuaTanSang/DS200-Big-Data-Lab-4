import idx2numpy
import socket
import json
import time
from tqdm import tqdm

class DatasetStreamer:
    def __init__(self, image_path, label_path, host='localhost', port=6100):
        self.images = idx2numpy.convert_from_file(image_path)
        self.labels = idx2numpy.convert_from_file(label_path)
        assert len(self.images) == len(self.labels), "Mismatch images/labels"
        self.host = host
        self.port = port
        self.conn = None

    def start_server(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        print(f"[Streamer] Waiting for connection on {self.host}:{self.port} …")
        self.conn, addr = srv.accept()
        print(f"[Streamer] Connected by {addr}")
        srv.close()

    def stream_epochs(self, batch_size=100, sleep_time=2, epochs=1):
        total_batches = len(self.images) // batch_size
        # 1) accept connection once
        self.start_server()

        # 2) loop qua các epoch
        for ep in range(epochs):
            print(f"[Streamer] Starting epoch {ep+1}/{epochs}")
            for i in tqdm(range(total_batches), desc=f"[Ep {ep+1}]"):
                imgs = self.images[i*batch_size:(i+1)*batch_size]
                lbls = self.labels[i*batch_size:(i+1)*batch_size]
                payload = {
                    idx: {
                        "features": img.flatten().astype(float).tolist(),
                        "label": int(lbl)
                    }
                    for idx, (img, lbl) in enumerate(zip(imgs, lbls))
                }
                try:
                    self.conn.sendall((json.dumps(payload) + "\n").encode())
                except BrokenPipeError:
                    print("[Streamer] Client disconnected early!")
                    return
                time.sleep(sleep_time)

        # 3) đóng kết nối sau khi hết all epochs
        self.conn.close()
        print("[Streamer] Finished all epochs, connection closed.")
