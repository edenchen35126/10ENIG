import os
import sys
import collections
# Temporarily comment out to allow error messages for debugging
# Suppress FFmpeg error messages
# sys.stderr = open(os.devnull, 'w')
# Set FFmpeg log level to quiet
# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"

import cv2
import queue
import time
import threading
from ultralytics import YOLO
import requests
import numpy as np
from collections import deque
import av

# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"

# Force detection to use CPU
device = "cpu"
# print(f"Using device: {device}")

# Alert settings
alert_threshold = 0.85  # Confidence threshold for alerts
alert_cooldown = 300  # Cooldown time in seconds
alert_records = []  # Store detection events for batch alerts
last_alert_times = {}  # Store the last alert time for each camera
alert_interval = 300  # Batch alert interval (5 minutes)

# 類別對應的冷卻時間
class_alert_cooldowns = {
        0: 300,  # 火光
        1: 300,  # 煙霧
        2: 300,
        3: 300,
        4: 300,
        5: 300,
        6: 300
    }

# Mutex lock to ensure thread safety
mutex = threading.Lock()

# Load YOLOv8 model (ensure this model is trained with the 7 specified classes)
# model_no_gloves = YOLO('model/gloves_goggles/best_0320.pt').to(0)  # Replace 'best.pt' with your model file
# model_cleaning_process = YOLO('model/cleaning_process/best_0320.pt').to(0)

# Mapping from class index to event name (English)
class_event_mapping_en = {
    0: "no gloves",
    1: "without goggles",
    2: "cramp",
    3: "hang board",
    4: "towel",
    5: "process_wrong",
    6: "fall"


}

# Mapping from class index to event name (Chinese)
class_event_mapping_cn = {
    0: "未戴手套",
    1: "未戴護目鏡",
    2: "夾架",
    3: "上料",
    4: "毛巾",
    5: "未依清潔流程作業",
    6: "人員倒臥"

}

color_dict ={
    0: (255,0,255), 
    1: (0,255,0), #綠
    2: (255,0,0),
    3: (0,0,255),
    4: (255,255,0),
    5: (255,255,255),
    6: (0,255,255)

}


last_alert_times = {}
camera_urls = {"205001":"rtsp://hikvision:Unitech0815!@10.20.55.20","205002":"rtsp://hikvision:Unitech0815!@10.20.55.21"}
# camera_urls = {"205001":"C:/Users/vincent-shiu/Web/RecordFiles/2025-03-03/demo.mp4"}

for i in camera_urls:
    if i not in last_alert_times:
        last_alert_times[i] = {}

camera_rois = {"205001":(250,130,410,280),"205002":(1075,25,1250,250)}


# 每支攝影機的最新幀記憶區
frame_deques = {cam: deque(maxlen=1) for cam in camera_urls.keys()}

# 模型初始化（GPU ID 視需要調整）
model_map = {
    "model_no_gloves": YOLO('model/gloves_goggles/best_yb_0716.pt').to(0),
    "model_fall": YOLO('model/fall/best_0430.pt').to(0),
    "model_cleaning_process": YOLO('model/cleaning_process/best_0514.pt').to(0)
    }


# 每個攝影機分別推論哪些模型
camera_models = {
    "205001": ["model_no_gloves","model_cleaning_process","model_fall"],
    "205002": ["model_no_gloves","model_cleaning_process","model_fall"]
    }

def location_modify(bbox):
    # 因抹布效果會因背景效果不好，此方法是盡可能泛化
    x1 , y1 , x2 , y2 = bbox
    x1 = x1 - 10
    x2 = x2 + 10
    y1 = y1 - 10
    y2 = y2 + 10

    if x1 <= 1:
        x1 = 1
    if x2 >= 1279:
        x2 = 1279
    if y1 <= 1:
        y1 = 1
    if y2 >= 719:
        y2 = 719
    bbox = x1 , y1 , x2 , y2

    return bbox

def calculate_iou(bbox1, bbox2):
    #算IOU
    # 計算交集
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # 交集區域寬高
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    intersection = inter_width * inter_height
    
    # 計算兩個 BBox 的面積
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)
    
    # 計算聯集 (Union)
    union = area1 + area2 - intersection
    
    # 防止除以 0
    if union == 0:
        return 0
    
    return intersection / union

def union_bboxes(union_bbox, cleaning_location):
    updated_location = []

    for bbox in cleaning_location:
        iou = calculate_iou(union_bbox, bbox)
        
        # 如果 IOU > 0，進行 Union
        if iou > 0:
            # 進行 BBox 合併（Union）
            new_x1 = min(union_bbox[0], bbox[0])
            new_y1 = min(union_bbox[1], bbox[1])
            new_x2 = max(union_bbox[2], bbox[2])
            new_y2 = max(union_bbox[3], bbox[3])
            union_bbox = (new_x1, new_y1, new_x2, new_y2)
        else:
            # 如果 IOU = 0，保留原本 BBox
            updated_location.append(bbox)

    # 最後將合併後的 BBox 放入清單
    updated_location.append(union_bbox)
    return updated_location

# def receive_frames(cam_id, url):
#     print(f"📡 Starting receiver for {cam_id}")
#     #cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
#     cap = cv2.VideoCapture(url)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     global frame_delay
#     frame_delay = 1 / fps  # 計算每一幀應該等待的時間 (秒)
#     while not stop_event.is_set():
#         ret, frame = cap.read()
#         time.sleep(frame_delay)
#         if ret:
#             frame_deques[cam_id].append(frame)
#         else:
#             print(f"⚠️ {cam_id} disconnected, retrying...")
#             time.sleep(1)
#     cap.release()
#     print(f"🛑 Receiver stopped for {cam_id}")

def receive_frames(cam_id, url):
    print(f"Starting receiver for {cam_id}")
    input_container = av.open(url, options={"hwaccel": "cuda"})

    for frame in input_container.decode(video=0):
        if stop_event.is_set():
            break
        img = frame.to_ndarray(format="bgr24")
        frame_deques[cam_id].append(img)
    input_container.close()
    print(f"Receiver stopped for {cam_id}")

def run_model(model, frame, detections,model_name,roi=None):
    #detections = []
    # 使用模型進行推論
    #print("after :",roi)
    if model_name == "model_cleaning_process":
        results = model.predict(frame, conf=0.3,verbose=False)
    else:
        results = model.predict(frame, conf=alert_threshold,verbose=False)

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            cls = int(box.cls[0])  # Get class index
            
            if model_name == "model_cleaning_process":
                # hang board
                if (confidence > 0.9) and (cls == 1):
                    # Get bounding box coordinates
                    x1_box, y1_box, x2_box, y2_box = map(int, box.xyxy[0])

                    adjusted_box = {
                            "model": model_name,  # 添加模型名稱
                            'xyxy': [x1_box, y1_box, x2_box, y2_box],
                            'conf': box.conf,
                            'cls': box.cls
                            }
                    detections.append(adjusted_box)
                elif (confidence > 0.3) and ((cls == 0) or (cls == 2)):
                    # Get bounding box coordinates
                    x1_box, y1_box, x2_box, y2_box = map(int, box.xyxy[0])

                    adjusted_box = {
                            "model": model_name,  # 添加模型名稱
                            'xyxy': [x1_box, y1_box, x2_box, y2_box],
                            'conf': box.conf,
                            'cls': box.cls
                            }

                    detections.append(adjusted_box)
            else:
                if confidence > alert_threshold:
                    # Get bounding box coordinates
                    x1_box, y1_box, x2_box, y2_box = map(int, box.xyxy[0])
                    
                    # 檢查該類別是否受 ROI 限制
                    if roi and cls == 1 and model_name == "model_no_gloves":  # 假設類別 1 (without_goggles) 只能在 ROI 偵測
                        
                        x1, y1, x2, y2 = roi
                        if not (x1 <= x1_box <= x2 and y1 <= y1_box <= y2 and 
                                x1 <= x2_box <= x2 and y1 <= y2_box <= y2):
                            continue  # 如果 bbox 不在 ROI 內，跳過

                    adjusted_box = {
                            "model": model_name,  # 添加模型名稱
                            'xyxy': [x1_box, y1_box, x2_box, y2_box],
                            'conf': box.conf,
                            'cls': box.cls
                            }

                    detections.append(adjusted_box)


def inference_worker(camera_index):
    models = camera_models.get(camera_index, [])

    dis_cleaning_count = 20000
    cleaning_count = 0
    cleaning_location = []
    # 記錄過去 3 秒內的 bbox（使用 deque 維持 FIFO 緩存）
    past_bboxes = {
        2: collections.deque(),  # cramp
        4: collections.deque()   # towel
    }
    MAX_TIME = 3  # 3 秒

    mistake_count = 0
    dis_mistake_count = 2000

    print(f"Starting to display camera {camera_index}")

    while not stop_event.is_set():

        if frame_deques[camera_index]:
            raw_frame = frame_deques[camera_index][-1]

            #preview_frame = frame.copy()
            frame = raw_frame.copy()  # 明確切乾淨
            preview_frame = raw_frame.copy()

            roi_box = None

            if frame is not None:
                
                frame_height, frame_width = frame.shape[:2]
                # print(f"Camera {camera_index} - Frame size: {frame_width}x{frame_height}")
                
                # Retrieve ROI for this camera
                if str(camera_index) in camera_rois:
                    roi = camera_rois[str(camera_index)]
                    x1, y1, x2, y2 = roi
                    # Validate ROI coordinates against frame size
                    x1 = max(0, min(x1, frame_width - 1))
                    x2 = max(0, min(x2, frame_width))
                    y1 = max(0, min(y1, frame_height - 1))
                    y2 = max(0, min(y2, frame_height))


                    if x1 >= x2 or y1 >= y2:
                        print(f"Invalid ROI for camera {camera_index}: {roi}. Skipping ROI processing.")
                        roi = None
                        roi_frame = frame
                    else:
                        # Draw ROI rectangle on the frame
                        roi_cleaning_frame = np.zeros_like(frame)
                        if camera_index == "205001":
                            roi_cleaning_frame[0:720, 0:800] = preview_frame[0:720, 0:800]
                        elif camera_index == "205002":
                            roi_cleaning_frame[0:720, 400:1280] = preview_frame[0:720, 400:1280]
                            
                        # resized_roi_cleaning_frame = cv2.resize(roi_cleaning_frame, (640, 400))
                        # cv2.imshow(f"{camera_index}_test",resized_roi_cleaning_frame)

                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     stop_event.set()

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                        roi_frame = frame
                        roi_box = (x1,y1,x2,y2)


                        # Optionally, mask out the area outside ROI for detection
                        # mask = frame.copy()
                        # mask[:y1, :] = 0
                        # mask[y2:, :] = 0
                        # mask[:, :x1] = 0
                        # mask[:, x2:] = 0
                        # roi_frame = mask
                else:
                    roi = None
                    roi_frame = frame  # If no ROI defined, use the whole frame

                detections = []  # Store detections for alerts
                location_info = {}
                cleaning_flag = False
                process_flag = True

                for m_name in models:
                    # print("camera_index :",camera_index)
                    # print("landmark_bounding :",landmark_bounding)
                    model = model_map[m_name]
                    #print(m_name)
                    if m_name == "model_cleaning_process":
                        run_model(model, preview_frame, detections,m_name,roi_box)
                    else:
                        run_model(model, preview_frame, detections,m_name,roi_box)

                #wrong_time = time.strftime("%Y-%m-%d_%H-%M-%S")
                for box in detections:
                    
                    model_name = box["model"]
                    # 畫bounding box在preview frame上
                    x1_box, y1_box, x2_box, y2_box = box['xyxy']

                    confidence = float(box['conf'][0])
                    cls = int(box['cls'][0])

                    if model_name == "model_no_gloves":
                        #wrong_time = time.strftime("%Y-%m-%d_%H-%M-%S")
                        #cv2.imwrite(f"original/{model_name}/{camera_index}_{wrong_time}_ori.jpg",preview_frame)

                        if cls == 0:
                            cls = 0
                            event_name_en = class_event_mapping_en.get(0, "Unknown Event")
                        elif cls == 1:
                            cls = 1
                            event_name_en = class_event_mapping_en.get(1, "Unknown Event")
                    elif model_name == "model_cleaning_process":

                        if cls == 0:
                            cls = 2
                            event_name_en = class_event_mapping_en.get(2, "Unknown Event")
                        elif cls == 1:
                            cls = 3
                            event_name_en = class_event_mapping_en.get(3, "Unknown Event")
                        elif cls == 2:
                            cls = 4
                            event_name_en = class_event_mapping_en.get(4, "Unknown Event")

                        if cls not in location_info:
                            location_info[cls] = []  # 若類別不存在，則建立列表

                        if (cls == 2) or (cls == 4) or (cls == 3):
                            location_info[cls].append([x1_box, y1_box, x2_box, y2_box])  # 存入 bbox 資訊
                    elif model_name == "model_fall":
                        #cv2.imwrite(f"original/{model_name}/{camera_index}_{wrong_time}_ori.jpg",preview_frame)
                        cls = 6
                        event_name_en = class_event_mapping_en.get(6, "Unknown Event")

                    cv2.rectangle(frame, (x1_box, y1_box), (x2_box, y2_box), color_dict[cls], 3)
                    label = f'{event_name_en} {confidence:.2f}'
                    cv2.putText(frame, label, (x1_box, y1_box - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_dict[cls], 3)
                    
                    # if (cls == 0) or (cls == 1) or (cls == 6):
                    #     #wrong_time = time.strftime("%Y-%m-%d_%H-%M-%S")
                    #     cv2.imwrite(f"result/{model_name}/{camera_index}_{wrong_time}_result.jpg",frame)


                keys = [key for key in location_info.keys() if key != 3]
                num_classes = len(keys)

                current_time = time.time()
                # 清除超過 3 秒的 bbox
                for key in (2, 4):
                    while past_bboxes[key] and current_time - past_bboxes[key][0][0] > MAX_TIME:
                        past_bboxes[key].popleft()  # 移除時間最久遠的 bbox
                
                # 更新 past_bboxes，存入當前 frame 偵測到的 bbox
                for key in (2, 4):
                    if key in location_info:
                        past_bboxes[key].append((current_time, location_info[key]))  # 記錄時間和 bbox

                
                ### 兩兩比較每個frame底下的bbox是否在附近
                for i in range(num_classes):
                    for j in range(i+1, num_classes):
                        class1, class2 = keys[i], keys[j]
                        
                        ##原始寫法，沒有加延遲
                        #bboxes1, bboxes2 = location_info[class1], location_info[class2]

                        # 當前 frame 的 bbox
                        bboxes1 = location_info.get(class1, [])
                        bboxes2 = location_info.get(class2, [])
                        # 過去 3 秒內的 bbox（已經排除過期的）
                        past_bboxes1 = [bbox for _, bboxes in past_bboxes[class1] for bbox in bboxes]
                        past_bboxes2 = [bbox for _, bboxes in past_bboxes[class2] for bbox in bboxes]

                        # 合併當前 frame 和過去 3 秒的 bbox
                        all_bboxes1 = bboxes1 + past_bboxes1
                        all_bboxes2 = bboxes2 + past_bboxes2

                        for bbox1 in all_bboxes1:
                            #增泛化
                            bbox1 = location_modify(bbox1)
                            for bbox2 in all_bboxes2:
                                #增泛化
                                bbox2 = location_modify(bbox2)

                                intersection = bbox_intersection_area(bbox1, bbox2)
                                area1, area2 = bbox_area(bbox1), bbox_area(bbox2)

                                # 計算 IoU (交集面積 / 聯集面積)
                                union = area1 + area2 - intersection
                                iou_percentage = (intersection / union) * 100 if union > 0 else 0
                                ### 判定有在做清潔的動作是依據毛巾跟夾架之間的IOU，如果大於指定數字則代表有在清潔，並把清潔的資訊傳入變數cleaning_location
                                if iou_percentage > 0:
                                    
                                    #cleaning_flag = True
                                    union_bbox = bbox_union(bbox1, bbox2)
                                    #檢查union_bbox 有沒有在現有的cleaning_location裡面，如果有，產生更大的聯集並更新cleaning_location
                                    if cleaning_location:
                                        cleaning_location = union_bboxes(union_bbox, cleaning_location)
                                    else:
                                        cleaning_location.append(union_bbox)
                                    #有在清潔就加變數cleaning_count，並且reset 變數dis_cleaning_count，dis_cleaning_count表示沒有清潔的動作就會開始扣減，有清潔會reset
                                    cleaning_count = cleaning_count + 1
                                    dis_cleaning_count = 20000

                                # 計算 交集面積 / 較大 bbox 面積
                                #overlap_percentage = (intersection / max(area1, area2)) * 100 if max(area1, area2) > 0 else 0
                ### 存入 hang_board 的資訊
                keys = [key for key in location_info.keys() if key == 3]
                num_classes = len(keys)

                # if num_classes > 0:
                #     intersection_flag = False
                
                ### 如果當下的frame有hang_board的資訊
                for g in range(num_classes):
                    class3 = keys[g]
                    bboxes3 = location_info[class3]
                    #iou_record = []

                    ### 兩兩比較，比較的是hang_board的bbox是否在之前存的清潔位置附近
                    for bbox3 in bboxes3:
                        iou_record = []
                        for bbox4 in cleaning_location:
                            intersection = bbox_intersection_area(bbox3, bbox4)
                            area1, area2 = bbox_area(bbox3), bbox_area(bbox4)

                            # 計算 IoU (交集面積 / 聯集面積)
                            union = area1 + area2 - intersection
                            iou_percentage = (intersection / union) * 100 if union > 0 else 0

                            #存入IOU資訊
                            iou_record.append(iou_percentage) ################
                        #print(iou_record)
                        #如果當個hang_board跟其他的cleaning_location bbox比較發現都小於某個數字，假定該hang_board沒有正確清潔
                        if all(x <= 2 for x in iou_record) and iou_record:
                            mistake_count = mistake_count + 1
                            dis_mistake_count = 2000
                            break


                #如果一開始就有偵測到hang_board，並且當下沒有任何的cleaning_location資訊，則算定是未正確清潔

                # for bbox in cleaning_location:
                #     x1 , y1 , x2 , y2 = bbox
                #     cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 3)
                #cv2.line(frame, (640,360), (690,360), (0,0,255), 3)

                if num_classes > 0:
                    if not cleaning_location:
                        mistake_count = mistake_count + 1
                        dis_mistake_count = 2000   
                else:
                    mistake_count = 0
                # if num_classes > 0:
                #     if intersection_flag == False:
                #         mistake_count = mistake_count + 1
                #         dis_mistake_count = 2000


                #倒數計時器，針對的是會去計算連續時間內，是否持續沒有清潔或是沒有錯誤清潔流程的情況發生
                dis_cleaning_count = dis_cleaning_count - 1
                dis_mistake_count = dis_mistake_count - 1

                #如果有偵測到清潔且有陸續偵測到，則改Flag資訊
                if (cleaning_count > 5) & (dis_cleaning_count > 1000):
                    #cleaning_flag = True
                    pass
                #如果先前有偵測到cleaning_count，但是持續一段時間後，會自動reset flag，並清空清潔位置資訊
                elif dis_cleaning_count <= -3000:
                    #cleaning_flag = False
                    cleaning_count = 0
                    cleaning_location = []

                #如果有偵測到錯誤清潔且有陸續偵測到，則改Flag資訊
                if (mistake_count > 30) & (dis_mistake_count > 1500):
                    process_flag = False
                    # 把資訊儲存，並send alert
                    if not process_flag:
                        adjusted_box = {
                                        "model": "model_process_wrong",  # 添加模型名稱
                                        'cls': 5,
                                        "process_wrong_frame": preview_frame

                                        }
                        detections.append(adjusted_box)
                #如果先前有偵測到mistake_count，但是持續一段時間後，會自動reset flag & count
                elif dis_mistake_count <= 1500:
                    mistake_count = 0
                    process_flag = True

                if detections:
                    # 將警報處理放入獨立的線程，以避免阻塞
                    alert_thread = threading.Thread(target=send_alert, args=(preview_frame.copy(), camera_index, detections), daemon=True)
                    alert_thread.start()
                
                #print(f"cleaning_flag : {cleaning_flag} , cleaning_count : {cleaning_count} , dis_cleaning_count : {dis_cleaning_count} , mistake_count : {mistake_count} , dis_mistake_count : {dis_mistake_count} , process_status : {'[合乎process]' if process_flag else '[流程錯誤]'}")
                
                # video.write(frame)  # 寫入影片

                # resized_frame = cv2.resize(frame, (640, 400))

                # cv2.imshow(f"{camera_index}",resized_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()

        else:
            time.sleep(0.01)
    
    # video.release()

def bbox_union(bbox1, bbox2):
    """
    計算兩個 bbox 的聯集範圍，並返回聯集 BBox 的座標。
    """
    x_left = min(bbox1[0], bbox2[0])
    y_top = min(bbox1[1], bbox2[1])
    x_right = max(bbox1[2], bbox2[2])
    y_bottom = max(bbox1[3], bbox2[3])

    return [x_left, y_top, x_right, y_bottom]

def bbox_intersection_area(bbox1, bbox2):
    """ 計算兩個 bbox 的交集面積 """
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0  # 無交集

    return (x_right - x_left) * (y_bottom - y_top)

def bbox_area(bbox):
    """ 計算 bbox 的面積 """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def send_alert(send_frame, camera_index, detections):
    """
    發送警報的函數，處理偵測到的事件並與 API 進行互動。
    detections 現在是一個包含字典的列表，每個字典包含 'xyxy', 'conf', 'cls'。
    """
    cooldown_updated_classes = set()  # 延後更新的冷卻清單

    current_time = time.time()
    cls_buffer_cooldown_dections = []
    #cls_buffer_cooldown = {}

    for box in detections:
        
        model_name = box["model"]
        if model_name == "model_process_wrong":
            
            cls = int(box['cls'])  # 取得類別索引
        else:
            confidence = float(box['conf'][0])
            cls = int(box['cls'])  # 取得類別索引
        
        if model_name == "model_no_gloves":
            if cls == 0:
                cls = 0
            else:
                cls = 1
        if model_name == "model_cleaning_process":
            continue

        if model_name == "model_fall":
            cls = 6

        # 取得該類別的冷卻時間
        cooldown_time = class_alert_cooldowns.get(cls, 300)
        # 初始化該類別的警報時間
        if cls not in last_alert_times[str(camera_index)]:
            last_alert_times[str(camera_index)][cls] = 0
        
        if (current_time - last_alert_times[str(camera_index)][cls]) <= cooldown_time:
            continue

        if current_time - last_alert_times[str(camera_index)][cls] > cooldown_time:
            # last_alert_times[str(camera_index)][cls] = current_time
            # print(f"Sending alert for camera {camera_index} to API!")

            if model_name == "model_no_gloves":
                if cls == 0:
                    box["cls"] = 0
                    cls_buffer_cooldown_dections.append(box)
                    event_name_en = class_event_mapping_en.get(0, "Unknown Event")
                else:
                    box["cls"] = 1
                    cls_buffer_cooldown_dections.append(box)
                    event_name_en = class_event_mapping_en.get(1, "Unknown Event")
            if (model_name == "model_process_wrong"):
                box["cls"] = 5
                cls_buffer_cooldown_dections.append(box)
            if model_name == "model_fall":
                box["cls"] = 6
                cls_buffer_cooldown_dections.append(box)
                event_name_en = class_event_mapping_en.get(6, "Unknown Event")

        if (model_name != "model_process_wrong"):
            x1, y1, x2, y2 = box['xyxy']
            cv2.rectangle(send_frame, (x1, y1), (x2, y2), color_dict[cls], 3)
            label = f'{event_name_en} {confidence:.2f}'
            cv2.putText(send_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_dict[cls], 3)
        # ✅ 延遲更新冷卻時間
        cooldown_updated_classes.add(cls)
    
    # ✅ 統一寫入冷卻時間
    for cls in cooldown_updated_classes:
        last_alert_times[str(camera_index)][cls] = current_time

    if cls_buffer_cooldown_dections:    
        for box in cls_buffer_cooldown_dections:
            model_name = box["model"]
            if model_name == "model_process_wrong":
                
                cls = int(box['cls'])  # 取得類別索引
            else:
                confidence = float(box['conf'][0])
                cls = int(box['cls'])  # 取得類別索引

            if camera_index == "205001" or camera_index == "205002":
                location = "十課IG站"
            else:
                location = f"未知位置"

            # 格式化檔名為 "1-2024-12-12_17-53-11.jpg"
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            formatted_filename = f"{camera_index}-{timestamp}.jpg"
            
            if model_name == "model_process_wrong":

                wrong_frame = box["process_wrong_frame"]
                success = cv2.imwrite(f"images/{formatted_filename}", wrong_frame)
            # 保存警報截圖
            else:
                success = cv2.imwrite(f"images/{formatted_filename}", send_frame)
            
            if success:
                print(f"Saved screenshot: {formatted_filename}")
            else:
                print(f"Failed to save screenshot: {formatted_filename}")

            # 準備 API 請求的數據（使用中文事件名稱）
            api_url = "https://eip.pcbut.com.tw/File/UploadYoloImage"
                    
            #join_cls = int(box['cls'][0])
            event_names = []

            if model_name == "model_no_gloves":
                if cls == 0:
                    event_name_cn = class_event_mapping_cn.get(0, "Unknown Event")
                else:
                    event_name_cn = class_event_mapping_cn.get(1, "Unknown Event")
            if model_name == "model_process_wrong":
                event_name_cn = class_event_mapping_cn.get(5, "Unknown Event")
            if model_name == "model_fall":
                event_name_cn = class_event_mapping_cn.get(6, "Unknown Event")

            event_names.append(event_name_cn)
            print(event_names)
            formatted_event_name = "；".join(event_names)

            camera_model = {
                "cameraId": camera_index,
                "location": location,
                "eventName": formatted_event_name,
                "eventDate": time.strftime("%Y-%m-%d %H:%M:%S"),
                "notes": f"{len(detections)} events detected with confidence > {alert_threshold}",
                "fileName": formatted_filename,
                "result": f"疑似發生 {formatted_event_name}, 請同仁儘速查看"
            }

            # 發送包含影像和攝影機數據的 POST 請求
            # "D:/My Documents/vincent-shiu/桌面/ENIG/images/"+

            try:
                with open(f"images/{formatted_filename}", 'rb') as img_file:
                    files = {'files': (formatted_filename, img_file, 'image/jpeg')}
                    response = requests.post(api_url, files=files, data=camera_model, verify=False)

                if response.status_code == 200:
                    print(f"Successfully sent alert for camera {camera_index}. Response: {response.text}")

                else:
                    print(f"Failed to send alert for camera {camera_index}. Status Code: {response.status_code}, Response: {response.text}")
                            

            except Exception as e:
                print(f"Error sending alert for camera {camera_index}: {e}")
    #print(last_alert_times)

# Global stop event for all threads
stop_event = threading.Event()

def batch_alert():
    """
    處理批次警報的函數，每隔一段時間檢查一次 alert_records 並發送報告。
    """
    while not stop_event.is_set():
        time.sleep(alert_interval)
        with mutex:
            if alert_records:
                alert_message = f"Alert Report: {len(alert_records)} events detected."
                print(alert_message)
                # 在此添加批次警報郵件發送邏輯
                alert_records.clear()



if __name__ == '__main__':
    # Start batch alert processing thread #daemon: 主程式結束時強制結束thread
    alert_thread = threading.Thread(target=batch_alert, daemon=True)
    alert_thread.start()

    # Start threads for all cameras
    camera_threads = []
    # for index, url in enumerate(camera_urls):
    #     threads = process_camera(index, url)
    #     camera_threads.extend(threads)

    # global video
    # # 讀取影片
    # cap = cv2.VideoCapture("C:/Users/vincent-shiu/Web/RecordFiles/2025-03-03/demo.mp4")
    # output_video = "C:/Users/vincent-shiu/Web/RecordFiles/2025-03-03/result.mp4"
    # # 確保影片成功讀取
    # if not cap.isOpened():
    #     print("無法讀取影片！")
    #     exit()
    # # 取得影片資訊
    # fps = int(cap.get(cv2.CAP_PROP_FPS))  # 幀率
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 影片寬度
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 影片高度
    # # 設定影片編碼格式與輸出物件
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 影片格式（mp4）
    # video = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))


    for index, (key,value) in enumerate(camera_urls.items()):
        camera_threads.append(threading.Thread(target=receive_frames, args=(key, value), daemon=True))
        camera_threads.append(threading.Thread(target=inference_worker, args=(key,), daemon=True))
        # threads = process_camera(key, value)
        # camera_threads.extend(threads)
    for t in camera_threads:
        t.start()

    try:
        # Wait for all threads to complete
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        print("Interrupt received, stopping all threads...")
    finally:
        for thread in camera_threads:
            thread.join(timeout=2)
        alert_thread.join(timeout=2)
        cv2.destroyAllWindows()
        print("All resources released, exiting.")