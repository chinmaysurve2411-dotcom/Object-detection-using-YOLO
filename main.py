import cv2
import time
from ultralytics import YOLO


class YOLOv8Detector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720
    ):
        """
        Initialize YOLOv8 detector.
        """

        print("[INFO] Loading YOLO model...")
        self.model = YOLO(model_path)

        print("[INFO] Starting webcam...")
        
        # DirectShow backend fixes MSMF webcam issues on Windows
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError(
                "[ERROR] Could not access webcam.\n"
                "Try changing the camera index or close other apps using the camera."
            )

        # Webcam resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # FPS calculation
        self.prev_frame_time = 0
        self.curr_frame_time = 0

        print("[INFO] Webcam started successfully.")
        print("[INFO] Press 'Q' to quit.\n")

    def calculate_fps(self) -> int:
        """
        Calculate FPS.
        """

        self.curr_frame_time = time.time()

        fps = 1 / (self.curr_frame_time - self.prev_frame_time) \
            if self.prev_frame_time != 0 else 0

        self.prev_frame_time = self.curr_frame_time

        return int(fps)

    def process_frame(self, frame):
        """
        Run YOLO inference on a frame.
        """

        results = self.model(frame)

        annotated_frame = results[0].plot()

        return annotated_frame

    def draw_fps(self, frame, fps: int):
        """
        Draw FPS on frame.
        """

        cv2.putText(
            frame,
            f"FPS: {fps}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    def run(self):
        """
        Main detection loop.
        """

        while True:
            success, frame = self.cap.read()

            if not success:
                print("[ERROR] Failed to read frame from webcam.")
                break

            # Resize frame for better performance
            frame = cv2.resize(frame, (1280, 720))

            # YOLO detection
            annotated_frame = self.process_frame(frame)

            # FPS
            fps = self.calculate_fps()
            self.draw_fps(annotated_frame, fps)

            # Show output
            cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

            # Exit on Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] Exiting application...")
                break

        self.cleanup()

    def cleanup(self):
        """
        Release resources.
        """

        self.cap.release()
        cv2.destroyAllWindows()

        print("[INFO] Webcam released.")
        print("[INFO] Application closed successfully.")


if __name__ == "__main__":
    detector = YOLOv8Detector(
        model_path="yolov8n.pt",
        camera_index=0
    )

    detector.run()
