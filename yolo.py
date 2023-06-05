import cv2
import numpy as np
import argparse
import time

garbage_list = [
    "bottle",
    "backpack",
    "umbrella",
    "handbag",
    "suitcase",
    "kite",
    "frisbee",
    "cup",
    "banana",
    "apple",
    "orange",
    "cell_phone",
    "book",
]
is_intersection = False
litter_done = False
parser = argparse.ArgumentParser()
parser.add_argument("--webcam", help="True/False", default=True)
parser.add_argument("--play_video", help="True/False", default=True)
parser.add_argument("--image", help="True/False", default=False)
parser.add_argument(
    "--video_path", help="Path of video file", default="videos/car_on_road.mp4"
)
parser.add_argument(
    "--image_path", help="Path of image to detect objects", default="Images/man.jpg"
)
parser.add_argument("--verbose", help="To print statements", default=True)
parser.add_argument("--save_video", help="To save video", default=True)
args = parser.parse_args()


def hog_detect(video_path):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cv2.startWindowThread()

    # open webcam video stream
    cap = cv2.VideoCapture(video_path)

    # the output will be written to output.avi
    out = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (640, 480)
    )

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # resizing for faster detection
        frame = cv2.resize(frame, (640, 480))
        # using a greyscale picture, also for faster detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # detect people in the image
        # returns the bounding boxes for the detected objects
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        for xA, yA, xB, yB in boxes:
            # display the detected boxes in the colour picture
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # Write the output video
        out.write(frame.astype("uint8"))
        # Display the resulting frame
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# Load yolo
def load_yolo():
    net = cv2.dnn.readNet("model_data/yolov3.weights", "cfg/yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def load_image(img_path):
    # image loading
    img = cv2.imread(img_path)
    print(img.size)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels


def start_webcam():
    cap = cv2.VideoCapture(0)

    return cap


def display_blob(blob):
    """
    Three images each for RED, GREEN, BLUE channel
    """
    for b in blob:
        for n, imgb in enumerate(b):
            cv2.imshow(str(n), imgb)


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=0.00392,
        size=(320, 320),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]

            if class_id == 0 or class_id == 39:
                # if class_id == 39 and conf > 0:
                # print(conf)
                if conf > 0.2:
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confs.append(float(conf))
                    class_ids.append(class_id)
    return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    global is_intersection
    global litter_done
    global garbage_list
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    personExist = False
    phoneExist = False
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "person":
                # print("Person Exists")
                personExist = True
                personImage = {"x1": x, "x2": x + w, "y1": y, "y2": y + h}
            if label in garbage_list:
                # print("Phone Exists")
                phoneExist = True
                phoneImage = {"x1": x, "x2": x + w, "y1": y, "y2": y + h}

            color = colors[i]
            if label == "person":
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
            # cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            # cv2.putText(img, label, (x, y - 5), font, 1, color, 1)

    if personExist and phoneExist:
        inter_val = get_intersection(personImage, phoneImage)
        if litter_done:
            if inter_val > 0 and phoneImage["y2"] > 350:
                print("Cleaner Detected")
                litter_done = False
                is_intersection = True

        if inter_val > 0:
            is_intersection = True

        if is_intersection == True:
            if inter_val == 0:
                for i in range(len(boxes)):
                    if str(classes[class_ids[i]]) != "person":
                        print("Littering ", {str(classes[class_ids[i]])})
                        break
                litter_done = True
                is_intersection = False

    cv2.imshow("Image", img)


def image_detect(img_path):
    model, classes, colors, output_layers = load_yolo()
    image, height, width, channels = load_image(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    draw_labels(boxes, confs, colors, class_ids, classes, image)
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break


def webcam_detect():
    model, classes, colors, output_layers = load_yolo()
    cap = start_webcam()
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()


def start_video(video_path):
    model, classes, colors, output_layers = load_yolo()
    print(video_path)
    # video_path = "manthan.mp4"
    cap = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_MSMF)
    result = cv2.VideoWriter(
        video_path + "_processed.avi",
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (int(cap.get(3)), int(cap.get(4))),
    )
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        key = cv2.waitKey(1)
        result.write(frame)
        if key == 27:
            break

    cap.release()


def get_intersection(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    return intersection_area


if __name__ == "__main__":
    webcam = args.webcam
    video_play = args.play_video
    image = args.image
    if webcam:
        if args.verbose:
            print("---- Starting Web Cam object detection ----")
        webcam_detect()
    if video_play:
        video_path = args.video_path
        if args.verbose:
            print("Opening " + video_path + " .... ")
        start_video(video_path)
    if image:
        image_path = args.image_path
        if args.verbose:
            print("Opening " + image_path + " .... ")
        image_detect(image_path)

    cv2.destroyAllWindows()
