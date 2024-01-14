import cv2
import numpy as np
import threading
import queue
import time

class Cam(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.camera = cv2.VideoCapture(0)
        self.queue = queue
        self.running = True

    def run(self):
        while self.running:
            _, image = self.camera.read()
            depth = self.get_depth_data()  # 通过你的方法获取深度信息
            data = {'image': image, 'depth': depth}
            self.queue.put(data)
            cv2.imshow("image", image)
            cv2.waitKey(1)
            # time.sleep(0.1)  # 控制图像读取的频率，根据实际需求调整

    def get_depth_data(self):
        # 这里添加获取深度信息的代码
        # 示例中使用随机生成的深度数据，实际中需要根据相机或传感器获取
        depth = np.random.rand(480, 640) * 100
        return depth

    def stop(self):
        self.running = False

class Grasp:
    def __init__(self, queue):
        self.queue = queue

    def calculate_grasp_pose(self):
        while True:
            if not self.queue.empty():
                time.sleep(4)  # 控制计算频率，根据实际需求调整
                data = self.queue.get()
                image = data['image']
                depth = data['depth']
                # 这里添加抓取姿态计算的代码，使用image和depth
                # 返回变换矩阵 T
                T = np.eye(4)  # 示例中使用单位矩阵
                return T

class Move:
    def execute_grasp(self, T):
        # 这里添加执行抓取动作的代码
        time.sleep(1)
        print(f"Executing grasp with transformation matrix: {T}")

class Publish:
    def publish_data(self, image, depth, T):
        # 这里添加将图像、深度和矩阵发布在局域网中的代码
        print("Publishing data to the local network")

# 测试
if __name__ == "__main__":
    data_queue = queue.Queue()

    cam = Cam(data_queue)
    grasp = Grasp(data_queue)
    move = Move()
    publish = Publish()

    cam.start()

    try:
        while True:
            # 计算抓取姿态
            T = grasp.calculate_grasp_pose()

            # 执行抓取动作
            move.execute_grasp(T)

            # 发布数据
            data = data_queue.get()
            publish.publish_data(data['image'], data['depth'], T)

            time.sleep(0.1)  # 控制循环频率，根据实际需求调整

    except KeyboardInterrupt:
        cam.stop()
        cam.join()
