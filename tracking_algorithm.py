import json
import cv2
import colorsys
import numpy as np
from munkres import Munkres, print_matrix # pip install munkres


# Tracking class
class Tracker():
    next_id = 0
    
    def __init__(self, class_id, object_id, point = []):
        self.tracker_id = Tracker.next_id
        Tracker.next_id += 1
        self.class_id = class_id
        self.object_id = object_id
        self.point = point # [x1, y1, x2, y2]
    # init tracker_id
    def clear_id(self):
        Tracker.next_id = 0

# annotation dictionary, previous image, present image
# return = matrix(row = previous image object_id, column = present image object_id, value = cost)
def matrix_create(annotation, previous, now):
    G = list()
    for p in annotation[previous]:
        g = []
        for q in annotation[now]:
            loss = abs(p['x'] - q['x']) + abs(p['y'] - q['y'])
            g.append(loss)
        G.append(g)
    
    return G

# load detection file
'''
json format
dataset['labels'] = [{
    'image':'000000.jpg',
    'point':{'x':242, 'y':122},
    'width': 12.42,
    'height': 15.2,
    'cls_id':4,
    'object_id':0
}]
'''
with open('./demo/drive6_final2.json', 'r') as f:
    data = json.load(f)

# create annoation dictionary
annotation = dict()
for anno in data['labels']:
    if anno['image'] not in annotation:
        annotation[anno['image']] = []
        annotation[anno['image']].append({'x':anno['point']['x'], 'y':anno['point']['y'], 'w': anno['width'], 'h': anno['height'], 'class':anno['cls_id'], 'Object_id':anno['Object_id']})
    else:
        annotation[anno['image']].append({'x':anno['point']['x'], 'y':anno['point']['y'], 'w': anno['width'], 'h': anno['height'], 'class':anno['cls_id'], 'Object_id':anno['Object_id']})

# all class
class_label_names = ('background', # zero class
                    'Slow', 'Stop', 'yield', 'noStopping', 'noParking', 'noGoingStraight',
                    'noRightTurn', 'noLeftTurn', 'noUTurn', 'speedLimit60', 'speedLimit50',
                    'Rotary', 'AheadOnly', 'TurnRight', 'TurnLeft', 'goStraightOrRight',
                    'goStraightOrLeft', 'UTurn', 'CrosswalkSign', 'ChildrenSign', 'OnewayRight',
                    'OnewayLeft', 'OnewayStraight', 'yieldToOncomingTraffic', 'railroadCrossing',
                    'signalFlag', 'Crosswalk_yellow', 'ChildrenSign_yellow', 'bicycle', 'tunnel',
                    'bridge', 'slipperyRoad', 'roughRoad', 'speedBump', 'otherCautionarySigns',
                    'noThoroughfare', 'noCar', 'noFreightCar', 'noVan', 'noMotorcycle', 'noMotorcycleAndCar',
                    'noCultivator', 'noBicycle', 'noEntry', 'noOutstripping', 'weightLimit', 'carHeightLimit',
                    'carWidthLimit', 'securingDistancebtwCar', 'lowestSpeedLimit', 'noPedestrianWalking',
                    'noDangerousGoodsVehicle', 'otherProhibitedSigns', 'TurnLeftorUTurn', 'TurnRightorLeft',
                    'TwowayTraffic', 'RightSidePass', 'LeftSidePass', 'PassageClassification', 'Bypass',
                    'PedestrianOnlyRoad', 'ElderlyCareSign', 'ProtectionOfTheDisabled', 'OtherIndicators',
                    'TrafficLight')


'''
version = 4
modify
- update point modify
'''

path = './demo/image/' # image path
distance = 90 # cost threshold
tracker_list = []

# main for loop (length = image amount)
for i in range(1000):
    if i != 0:
        previous_image = '{:06d}.jpg'.format(i-1)
        if previous_image not in annotation:
            continue
    image = '{:06d}.jpg'.format(i)
    if image not in annotation:
        continue
    
    G = matrix_create(annotation, previous_image, image)
    
    # hungarian_algorithm
    m = Munkres()
    indexes = m.compute(G) # hungarian result

    # create tracker_object id list
    track_obj = []
    for tidx, tracker in enumerate(tracker_list):
        track_obj.append(tracker.object_id)
    
    # create previous object_id list
    row_list = []
    for row, column in indexes:
        value = G[row][column]
        if value < distance:
            row_list.append(row)
            
    # tracking object delete
    remove = []
    for tracker in tracker_list:
        if tracker.object_id not in row_list:
            remove.append(tracker.object_id)

    for r in remove:
        for tidx, tracker in enumerate(tracker_list):
            if tracker.object_id == r:
                del tracker_list[tidx]

     # create or update tracker       
    for row, column in indexes:
        value = G[row][column]
        # create tracker
        if value < distance:
            if row not in track_obj:
                for anno in annotation[previous_image]:
                    if anno['Object_id'] == row:
                        class_id = anno['class']
                tracker_list.append(Tracker(class_id, column))
            # update tracker
            else:
                for tracker in tracker_list:
                    if tracker.object_id == row:
                        tracker.object_id = column
    
    # bounding box visualization
    img = cv2.imread(path + image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # color
    num_classes = 20
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    
    for tidx, tracker in enumerate(tracker_list):
        try:
            a = annotation[image][tracker.object_id]
        except Exception as e:
            print(previous_image, image)
            print(e)
        x1 = a['x'] - 1/2 * a['w']
        x2 = a['x'] + 1/2 * a['w']
        y1 = a['y'] - 1/2 * a['h']
        y2 = a['y'] + 1/2 * a['h']
        tracker.point = [x1, y1, x2, y2] # update point

        # drawing bouding box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colors[tracker.class_id % num_classes], 2)
        cv2.rectangle(img, (int(x1), int(y1-30)), (int(x1+len(str(tracker.tracker_id))*17), int(y1)), colors[tracker.class_id % num_classes], -1)
        cv2.putText(img, str(tracker.tracker_id),(int(x1), int(y1-10)),0, 0.75, (255,255,255),2)
        cv2.putText(img, class_label_names[tracker.class_id] + '_' + str(tracker.tracker_id), (10, 20 + 25*tidx), 0, 0.6, colors[tracker.class_id % num_classes], 2)
    
    result = np.asarray(img)
    result = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # save image
    cv2.imwrite('./demo/tracking/' + image, result)
    print('save image {}'.format(image))