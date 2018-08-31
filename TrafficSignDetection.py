import numpy as np
import cv2

import os
import DetectionMethod
import DetectionMethodWeather
import DetectionMethodDark


DIR = "databases/input"
OUTPUT_DIR = "databases/output"



def no_challenge_detection():
    video_dir_list = [f for f in os.listdir(DIR)]
    video_dir_list.sort()
    #print video_dir_list

    img_list = []

    for video_dir in video_dir_list:
        path = os.path.join(DIR, video_dir)

        for f in os.listdir(os.path.join(path)):
            if f.endswith('.png'):
                img_list.append((video_dir,f))



    #print len(img_list)
    for img in img_list:

        if img[0][6:8] == "00" or img[0] == "lisa":
        #if img[0]== "01_04_00_00_00":
             #if img[0] == "01_37_00_00_00":
            #print img[0]
            path = "databases/input/{}/{}".format(img[0], img[1])



            results = DetectionMethod.traffic_sign_detection(img)
            if results != {}:
                print "------------FRAME:{}----------------".format(img[1])
                print results
            image = cv2.imread(path)
            output = mark_traffic_signs(image, results)

        if not os.path.exists("databases/output/{}".format(img[0])):
            #print "create"
            os.makedirs("databases/output/{}".format(img[0]))

        cv2.imwrite("databases/output/{}/{}.png".format(img[0],img[1]), output)


def mark_traffic_signs(image_in, signs_dict):
    """Marks the center of a traffic sign and adds its coordinates.

    This function uses a dictionary that follows the following
    structure:
    {'sign_name_1': (x, y), 'sign_name_2': (x, y), etc.}

    Where 'sign_name' can be: 'stop', 'no_entry', 'yield',
    'construction', 'warning', and 'traffic_light'.

    Use cv2.putText to place the coordinate values in the output
    image.

    Args:
        image_in (numpy.array): the image to draw on.
        signs_dict (dict): dictionary containing the coordinates of
        each sign found in a scene.

    Returns:
        numpy.array: output image showing markers on each traffic
        sign.
    """
    output = np.copy(image_in)
    if signs_dict is not None:

        for key in signs_dict:
            pos = signs_dict[key]
            #print pos
            for p in pos:
                x, y, w, h = p
                #print "{} at {}".format(key, p)

                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
                text = "%s" % (key)
                cv2.putText(output, text, (x-5, y+h+15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                #cv2.putText(output, text, (100,400), cv2.FONT_HERSHEY_PLAIN, 20, (0, 255, 0), 20)
    return output


def bad_weather_detection():
    video_dir_list = [f for f in os.listdir(DIR)]
    video_dir_list.sort()
    #print video_dir_list

    img_list = []

    for video_dir in video_dir_list:
        path = os.path.join(DIR, video_dir)

        for f in os.listdir(os.path.join(path)):
            if f.endswith('.png'):
                img_list.append((video_dir,f))



    #print len(img_list)
    for img in img_list:

        if img[0][6:11] == "01_11":
                #and img[1] == "01_45_01_11_02_102.png":
            #print img[0]
            path = "databases/input/{}/{}".format(img[0], img[1])



            results = DetectionMethodWeather.traffic_sign_detection(img)
            if results != {}:
                print "------------FRAME:{}----------------".format(img[1])
                print results
            image = cv2.imread(path)
            output = mark_traffic_signs(image, results)

            if not os.path.exists("databases/output/{}".format(img[0])):
                #print "create"
                os.makedirs("databases/output/{}".format(img[0]))

            cv2.imwrite("databases/output/{}/{}.png".format(img[0],img[1]), output)




def dark_detection():
    video_dir_list = [f for f in os.listdir(DIR)]
    video_dir_list.sort()
    # print video_dir_list

    img_list = []

    for video_dir in video_dir_list:
        path = os.path.join(DIR, video_dir)

        for f in os.listdir(os.path.join(path)):
            if f.endswith('.png'):
                img_list.append((video_dir, f))

    #print len(img_list)
    for img in img_list:


        if img[0][6:11] == "01_04":
            # print img[0]
            path = "databases/input/{}/{}".format(img[0], img[1])



            results = DetectionMethodDark.traffic_sign_detection(img)
            if results != {}:
                print "------------FRAME:{}----------------".format(img[1])
                print results
            image = cv2.imread(path)
            output = mark_traffic_signs(image, results)

            #if not os.path.exists("databases/output/{}".format(img[0])):
                # print "create"
                #os.makedirs("databases/output/{}".format(img[0]))

            cv2.imwrite("databases/output/{}/{}.png".format(img[0], img[1]), output)

if __name__ == '__main__':
    no_challenge_detection()
    bad_weather_detection()
    dark_detection()

