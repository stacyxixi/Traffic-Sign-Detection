import cv2
import os
import numpy as np
import math


def preprocessing(path):

    image = cv2.imread(path)
    #print image.shape

    #output_path = "output/test/stop"

    img_blur00= np.copy(image)
    img_blur0 = cv2.GaussianBlur(image, (5,5),0)
    img_blur1 = cv2.blur(image, (7,7))
    img_blur2 = cv2.bilateralFilter(image,5,150,150)

    return img_blur2


def traffic_sign_detection(img):
    """Finds all traffic signs in a real image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction


    Args:
        img (tuple): input image folder name, input image name.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """

    path = "databases/input/{}/{}".format(img[0], img[1])
    img_processed = preprocessing(path)
    #output_path = "output/test/stop"
    #cv2.imwrite(os.path.join(output_path,"{}_blur.png".format(img[1][0:17])),img_processed)

    results = {}


    result_red = classify_ROI_red(img_processed, img)
    results = result_red

    radii_range = range(10, 30, 1)
    results_light = traffic_light_detection(img_processed, img, radii_range)
    if results_light != {}:
        #results = results_light
        if results_light.has_key('red_light'):
            results['red_light'] = results_light['red_light']
        if results_light.has_key('green_light'):
            results['green_light'] = results_light['green_light']

    results_warning_construction = warning_construction_sign_detection(img_processed, img)
    if results_warning_construction != {}:

        if results_warning_construction.has_key('warning'):
            results['warning'] = results_warning_construction['warning']
        if results_warning_construction.has_key('construction'):
            results['construction'] = results_warning_construction['construction']

    return results

def generate_red_mask(image):

    red_low_1 = (0, 50, 20)
    red_low_2 = (10, 255, 255)
    red_high_1 = (170, 50, 20)
    red_high_2 = (180, 255, 255)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask_low = cv2.inRange(hsv, np.array(red_low_1, dtype="uint8"), np.array(red_low_2, dtype="uint8"))
    red_mask_high = cv2.inRange(hsv, np.array(red_high_1, dtype="uint8"), np.array(red_high_2, dtype="uint8"))
    image_red = cv2.addWeighted(red_mask_low, 1.0, red_mask_high, 1.0, 0.0)

    return image_red

def classify_ROI_red(img_in, img):

    results = {}
    pos_s = []
    pos_d = []
    pos_y = []

    roi_list, pos_list = detect_ROI_red(img_in, img)
    #print pos_list

    roi_test_path = "output/test/stop/{}".format(img[1][0:17])
    #print roi_test_path


    template1 = cv2.imread("databases/template/stop1.png")
    template2 = cv2.imread("databases/template/doNotEnter1.png")
    template3 = cv2.imread("databases/template/yield.bmp")

    temp1_red = generate_red_mask(template1)
    temp2_red = generate_red_mask(template2)
    temp3_red = generate_red_mask(template3)

    #cv2.imwrite("databases/template/stop.png", temp1_red)
    #cv2.imwrite("databases/template/doNotEnter.png", temp2_red)
    #cv2.imwrite("databases/template/yield.png", temp3_red)


    result1 = evaluate_template_match(roi_list,temp1_red)
    id1, val1, p1 = result1
    #print "stopsign match at Roi{} position{} at value{}".format(id1,p1,val1)

    result2 = evaluate_template_match(roi_list,temp2_red)
    id2, val2, p2 = result2
    #print "donotEnterSign match at Roi{} position{} at value{}".format(id2, p2, val2)

    result3 = evaluate_template_match(roi_list,temp3_red)
    id3, val3, p3 = result3
    #print "yieldSign match at Roi{} position{} at value{}".format(id3, p2, val3)

    if val2 > 0.3 and val2 == max(val2, (val1 + 0.14), val3):
        pos2 = pos_list[id2]
        pos_d.append(pos2)
        results['doNotEnter'] = pos_d

    if val1 > 0.3 and val1 + 0.14 == max(val2, (val1 + 0.14), val3):
        pos1 = pos_list[id1]
        pos_s.append(pos1)
        results['stop'] = pos_s

    if val3 > 0.4 and val3 == max((val1+0.14), val2, val3):
        pos3 = pos_list[id3]
        pos_y.append(pos3)
        results['yield'] = pos_y

    return results

def evaluate_template_match(roi_list, template):

    res_max = 0
    id_max = 0
    loc_max = 0, 0

    for i in range(len(roi_list)):
        roi = roi_list[i]
        row = roi.shape[0]
        col = roi.shape[1]
        row_t = template.shape[0]
        col_t = template.shape[1]

        """
        scale = min(row/float(row_t), col/float(col_t))
        template_scale = cv2.resize(template,None, fx=scale, fy=scale)
        res = cv2.matchTemplate(roi, template_scale, cv2.TM_CCOEFF_NORMED)
        """
        scale = min(float(row_t)/row, float(col_t)/col)
        roi_scale = cv2.resize(roi, None, fx=scale, fy=scale)
        res = cv2.matchTemplate(template, roi_scale,cv2.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        #print i, min_val, max_val,min_loc, max_loc, scale, roi.shape,template.shape
        if max_val > res_max:
            res_max = max_val
            id_max = i
            loc_max = max_loc

    return id_max,res_max, loc_max

def evaluate_template_match_single(roi, template):


    row = roi.shape[0]
    col = roi.shape[1]
    row_t = template.shape[0]
    col_t = template.shape[1]

    """
    scale = min(row/float(row_t), col/float(col_t))
    template_scale = cv2.resize(template,None, fx=scale, fy=scale)
    res = cv2.matchTemplate(roi, template_scale, cv2.TM_CCOEFF_NORMED)
    """
    scale = min(float(row_t)/row, float(col_t)/col)
    roi_scale = cv2.resize(roi, None, fx=scale, fy=scale)
    res = cv2.matchTemplate(template, roi_scale,cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    return max_val, max_loc

def detect_ROI_red(img_in, img):


    red_low_1 = (0,50,50)
    red_low_2 = (5,255,255)
    red_high_1 = (175,50,50)
    red_high_2 = (180,255,255)

    img_draw = np.copy(img_in)
    img_copy = np.copy(img_in)
    img_hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    red_mask_low = cv2.inRange(img_hsv, np.array(red_low_1, dtype="uint8"), np.array(red_low_2, dtype="uint8"))
    red_mask_high = cv2.inRange(img_hsv, np.array(red_high_1, dtype="uint8"), np.array(red_high_2, dtype="uint8"))
    red_mask = cv2.addWeighted(red_mask_low, 1.0, red_mask_high, 1.0, 0.0)
    result_hsv = cv2.bitwise_and(img_in, img_in, mask=red_mask)

    images_path = "output/test/stop/{}".format(img[1][0:17])

    contours = cv2.findContours(red_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    #print len(contours)

    roi_list = []
    pos_list = []

    con_id = 0
    if contours is not None:
        for c in contours:
            con_id +=1
            con_area = cv2.contourArea(c)
            if con_area > 40:

                cv2.drawContours(img_draw, [c], -1, (0, 255, 0), 1)
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 255), 1)

                roi_red = red_mask[y: y+h, x: x+w]
                num_white_pixel = np.sum(roi_red) / 255
                num_total_pixel = w * h
                percentage = float(num_white_pixel)/num_total_pixel
                ratio = float(h)/w
                #print ratio

                #print "contour{}: Area:{},white_pixels{},percentage{}".format(con_id, con_area,num_white_pixel, percentage)

                if percentage > 0.1 and 0.33 < ratio < 3:
                    cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    #roi = img_copy[y: y + h, x:x + w]
                    size = max(w,h)
                    roi_red_square = red_mask[y: y + size, x: x + size]
                    roi_list.append(roi_red_square)
                    pos_list.append((x, y, size, size))

                #pos_list.sort(key = lambda x:x[2]*x[3], reverse=True)



    #cv2.imwrite("output/test/stop/{}_red_mask.png".format(img[1][0:18]), red_mask)
    #cv2.imwrite("output/test/stop/{}_hsv.png".format(img[1][0:18]), result_hsv)
    #cv2.imwrite("output/test/stop/{}_box.png".format(img[1][0:18]), img_draw)

    roi_id = 0


    #if not os.path.exists(images_path):
        #print roi_id
        #os.makedirs(images_path)
    #print "There are {} ROI".format(len(roi_list))

    for roi in roi_list:
        roi_id +=1
        #print roi.shape
        #cv2.imwrite(os.path.join(images_path, "red_roi_{}.png".format(roi_id)), roi)

    return (roi_list, pos_list)

def traffic_light_detection(img_in, img, radii_range):
    results = {}
    pos_red_positive = []
    pos_green_positive = []

    roi_list_red, pos_list_red = red_light_detection(img_in, img,radii_range)
    roi_list_green, pos_list_green = green_light_detection(img_in, img,radii_range)

    template_red = cv2.imread("databases/template/redlight.png")
    template_green = cv2.imread("databases/template/greenlight.png")

    temp_red = cv2.bilateralFilter(template_red,5,150,150)
    temp_green = cv2.bilateralFilter(template_green, 5, 150, 150)

    #cv2.imwrite("databases/template/red_light_blur.png", temp_red)
    #cv2.imwrite("databases/template/green_light_blur.png", temp_green)

    for id in range(len(roi_list_red)):

        roi_red = roi_list_red[id]
        pos_red = pos_list_red[id]

        result_red = evaluate_template_match_single(roi_red,temp_red)
        val1, p1 = result_red
        #print "redlight match at redRoi{} position{} at value{}".format(id,p1,val1)

        if val1 > 0.5:
            pos_red_positive.append(pos_red)
            results['red_light'] = pos_red_positive

    for id in range(len(roi_list_green)):

        roi_green = roi_list_green[id]
        pos_green = pos_list_green[id]

        result_green = evaluate_template_match_single(roi_green, temp_green)
        val2, p2 = result_green
        #print "greenlight match at greenRoi{} position{} at value{}".format(id, p2, val2)

        if val2 > 0.5:
            pos_green_positive.append(pos_green)
            results['green_light'] = pos_green_positive

    #result2 = evaluate_template_match(roi_list_green,temp_green)
    #id2, val2, p2 = result2
    #print "greenlight match at Roi{} position{} at value{}".format(id2, p2, val2)

    return results

def red_light_detection(img_in, img, radii_range):


    red_low_1 = (0, 180, 100)
    red_low_2 = (10, 255, 255)
    red_high_1 = (170, 180, 100)
    red_high_2 = (180, 255, 255)

    img_draw = np.copy(img_in)
    img_copy = np.copy(img_in)
    img_hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    red_mask_low = cv2.inRange(img_hsv, np.array(red_low_1, dtype="uint8"), np.array(red_low_2, dtype="uint8"))
    red_mask_high = cv2.inRange(img_hsv, np.array(red_high_1, dtype="uint8"), np.array(red_high_2, dtype="uint8"))
    red_mask = cv2.addWeighted(red_mask_low, 1.0, red_mask_high, 1.0, 0.0)

    images_path = "output/test/lights/{}".format(img[1][0:])

    contours = cv2.findContours(red_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    # print len(contours)

    roi_list = []
    pos_list = []

    con_id = 0
    if contours is not None:
        for c in contours:
            con_id += 1
            con_area = cv2.contourArea(c)
            if 400 > con_area > 20:

                cv2.drawContours(img_draw, [c], -1, (0, 255, 0), 1)
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 255), 1)
                #print w*h

                roi_red = red_mask[y: y + h, x: x + w]
                num_white_pixel = np.sum(roi_red) / 255
                num_total_pixel = w * h
                percentage = float(num_white_pixel) / num_total_pixel
                ratio = float(h) / w
                # print ratio

                # print "contour{}: Area:{},white_pixels{},percentage{}".format(con_id, con_area,num_white_pixel, percentage)

                if percentage > 0.1 and 0.5 < ratio < 2:
                    cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    # roi = img_copy[y: y + h, x:x + w]
                    # size = max(w, h)
                    roi_red_light = img_copy[y: y + 3* h, x: x + w]
                    roi_list.append(roi_red_light)
                    pos_list.append((x, y, w, 3*h))

                # pos_list.sort(key = lambda x:x[2]*x[3], reverse=True)

    #cv2.imwrite("output/test/lights/{}_red_mask.png".format(img[1][0:]), red_mask)
    #cv2.imwrite("output/test/lights/{}_box.png".format(img[1][0:]), img_draw)

    roi_id = 0

    #if not os.path.exists(images_path):
        # print roi_id
        #os.makedirs(images_path)
    # print "There are {} ROI".format(len(roi_list))

    for roi in roi_list:
        roi_id += 1
        #print roi.shape
        #cv2.imwrite(os.path.join(images_path, "red_roi_{}.png".format(roi_id)), roi)

    return roi_list, pos_list

def green_light_detection(img_in, img, radii_range):


    green_low = (80, 100, 100)
    green_high = (90, 255, 255)


    img_draw = np.copy(img_in)
    img_copy = np.copy(img_in)
    img_hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(img_hsv, np.array(green_low, dtype="uint8"), np.array(green_high, dtype="uint8"))

    images_path = "output/test/lights_green/{}".format(img[1][0:])

    contours = cv2.findContours(green_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    # print len(contours)

    roi_list = []
    pos_list = []

    con_id = 0
    if contours is not None:
        for c in contours:
            con_id += 1
            con_area = cv2.contourArea(c)
            if 400 > con_area > 20:

                cv2.drawContours(img_draw, [c], -1, (0, 255, 0), 1)
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 255), 1)
                #print w*h

                roi_red = green_mask[y: y + h, x: x + w]
                num_white_pixel = np.sum(roi_red) / 255
                num_total_pixel = w * h
                percentage = float(num_white_pixel) / num_total_pixel
                ratio = float(h) / w
                # print ratio

                # print "contour{}: Area:{},white_pixels{},percentage{}".format(con_id, con_area,num_white_pixel, percentage)

                if percentage > 0.1 and 0.5 < ratio < 2:
                    cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    # roi = img_copy[y: y + h, x:x + w]
                    # size = max(w, h)
                    y1 = max(y-2*h,1)
                    roi_green_light = img_copy[y1: y + h, x: x + w]
                    roi_list.append(roi_green_light)
                    pos_list.append((x, y1, w, 3*h))

                # pos_list.sort(key = lambda x:x[2]*x[3], reverse=True)

    #cv2.imwrite("output/test/lights_green/{}_green_mask.png".format(img[1][0:]), green_mask)
    #cv2.imwrite("output/test/lights_green/{}_box.png".format(img[1][0:]), img_draw)

    roi_id = 0

    #if not os.path.exists(images_path):
        # print roi_id
        #os.makedirs(images_path)
    # print "There are {} ROI".format(len(roi_list))

    for roi in roi_list:
        roi_id += 1
        #print roi.shape
        #cv2.imwrite(os.path.join(images_path, "green_roi_{}.png".format(roi_id)), roi)

    return roi_list, pos_list

def generate_yellow_mask(image, para1,para2):
    low = para1
    high = para2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_yellow = cv2.inRange(hsv, np.array(low, dtype="uint8"), np.array(high, dtype="uint8"))


    return image_yellow

def warning_construction_sign_detection(img_in, img):
    results = {}
    pos_warn_positive = []
    pos_cons_positive = []
    yellow_low = (15, 100, 50)
    yellow_high = (25, 255, 255)

    orange_low = (3, 100, 100)
    orange_high = (10, 255, 255)

    roi_list_warn, pos_list_warn = detect_ROI_yellow_orange(img_in, img, yellow_low, yellow_high, 1)
    roi_list_cons, pos_list_cons = detect_ROI_yellow_orange(img_in, img, orange_low, orange_high, 2)

    template_warn = cv2.imread("databases/template/warning.png")
    template_cons = cv2.imread("databases/template/construction.png")

    #temp_warn = cv2.bilateralFilter(template_warn,5,150,150)

    temp_warn = generate_yellow_mask(template_warn, yellow_low, yellow_high)
    temp_cons = cv2.bilateralFilter(template_cons,5,150,150)

    #cv2.imwrite("databases/template/warning_yellow_hsv.png", temp_warn)
    #cv2.imwrite("databases/template/construction_blur.png", temp_cons)

    for id in range(len(roi_list_warn)):

        roi_warn = roi_list_warn[id]
        pos_warn = pos_list_warn[id]

        result_warn = evaluate_template_match_single(roi_warn, temp_warn)
        val1, p1 = result_warn
        #print "warning match at yellowRoi{} position{} at value{}".format(id, p1, val1)

        if val1 > 0.4:
            pos_warn_positive.append(pos_warn)
            results['warning'] = pos_warn_positive

    for id in range(len(roi_list_cons)):

        roi_cons = roi_list_cons[id]
        pos_cons = pos_list_cons[id]

        result_cons = evaluate_template_match_single(roi_cons, temp_cons)
        val2, p2 = result_cons
        #print "construction match at orangeRoi{} position{} at value{}".format(id, p2, val2)

        if val2 > 0.3:

            pos_cons_positive.append(pos_cons)
            results['construction'] = pos_cons_positive

    #print results
    return results

def detect_ROI_yellow_orange(img_in, img,para1,para2,switch):
    yellow_low = para1
    yellow_high = para2

    switch = switch

    img_draw = np.copy(img_in)
    img_copy = np.copy(img_in)
    img_hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(img_hsv, np.array(yellow_low, dtype="uint8"), np.array(yellow_high, dtype="uint8"))

    images_path = "output/test/warning/{}".format(img[1][0:])

    contours = cv2.findContours(yellow_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    # print len(contours)

    roi_list = []
    pos_list = []

    con_id = 0
    if contours is not None:
        for c in contours:
            con_id += 1
            con_area = cv2.contourArea(c)
            if  con_area > 100:

                cv2.drawContours(img_draw, [c], -1, (0, 255, 0), 1)
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 255), 1)
                # print w*h

                roi_yellow = yellow_mask[y: y + h, x: x + w]
                num_white_pixel = np.sum(roi_yellow) / 255
                num_total_pixel = w * h
                percentage = float(num_white_pixel) / num_total_pixel
                ratio = float(h) / w
                # print ratio

                # print "contour{}: Area:{},white_pixels{},percentage{}".format(con_id, con_area,num_white_pixel, percentage)

                if percentage > 0.1 and 0.5 < ratio < 2:
                    cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    # roi = img_copy[y: y + h, x:x + w]
                    size = max(w, h)

                    roi_BGR = img_copy[y: y + size, x: x + size]
                    roi_yellow = yellow_mask[y: y + size, x: x + size]

                    if switch == 1:
                        roi_list.append(roi_yellow)
                    if switch == 2:
                        roi_list.append(roi_BGR)
                    pos_list.append((x, y, size, size))

                # pos_list.sort(key = lambda x:x[2]*x[3], reverse=True)

    #cv2.imwrite("output/test/warning/{}_yellow_mask.png".format(img[1][0:]), yellow_mask)
    #cv2.imwrite("output/test/warning/{}_box.png".format(img[1][0:]), img_draw)

    roi_id = 0

    #if not os.path.exists(images_path):
        # print roi_id
        #os.makedirs(images_path)
    # print "There are {} ROI".format(len(roi_list))

    for roi in roi_list:
        roi_id += 1
        # print roi.shape
        #cv2.imwrite(os.path.join(images_path, "yellow_roi_{}.png".format(roi_id)), roi)

    return roi_list, pos_list

