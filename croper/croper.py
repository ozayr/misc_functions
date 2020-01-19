import cv2
import numpy as np
import math
from tkinter import filedialog
from tkinter import *


def put_text_on_image(instruction_bar, image_width, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 0.5
    while cv2.getTextSize(text, font, text_scale, 1)[0][0] > image_width:
        text_scale -= 0.1
    cv2.putText(instruction_bar, text, (20, 20), font, text_scale, (255, 255, 255), 1)
    return instruction_bar


def show(img, title="", time=-1):
    cv2.imshow(title, img)
    key = cv2.waitKey(time)
    cv2.destroyAllWindows()
    return key


select = False
temp = list()
state = list()


def select_region_elastic(image, sub_region_select=False):
    # convert image to RGB so that the selection boxes can show up if image is gray scale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    main_region_contour = []
    sub_region_contours = []
    sub_region_contour = []
    sub_region_circles = []
    sub_region_squares = []

    # ==========================================================================
    # create inatruction bar to show instruction info
    instruction_image = np.zeros((30, image.shape[1], 3), np.uint8)
    text = '[r]-reset [q]-quit'
    instruction_image = put_text_on_image(instruction_image, image.shape[1], text)

    # ==========================================================================

    # ============================================================================================================
    # MAIN REGION CROP
    # while loop that allows selection of a region in an image
    def select_main_region_callback(event, x, y, flags, param):
        # callback that will outline the region selected on the image
        # nb when done the last and first selection point will be joined to form a closed region
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(buffer_image, (x, y), 2, (0, 0, 255), 2)
            main_region_contour.append([[x, y]])
            cv2.drawContours(buffer_image, np.array([main_region_contour[-2:]]), -1, (0, 0, 255), 1)

    # create buffer image to reset
    buffer_image = image.copy()

    while 1 and not sub_region_select:

        window_name = "Main Region Select"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, select_main_region_callback)
        cv2.imshow(window_name, cv2.vconcat([buffer_image, instruction_image]))
        key = cv2.waitKey(1)

        if key == 13:
            print("selecting main region done....")
            cv2.destroyAllWindows()
            break

        elif key == ord('r'):
            print("selecting main region reset....")
            buffer_image = image.copy()
            main_region_contour = []
        elif key == ord('q'):
            print("selecting main region quit....")
            cv2.destroyAllWindows()
            return 0

    # create mask to mask out the selected region from original image
    mask = np.zeros(image.shape, np.uint8)
    if main_region_contour:
        cv2.drawContours(mask, np.array([main_region_contour]), -1, (255, 255, 255), -1)
        buffer_image = cv2.bitwise_and(image, mask)

    # =========================================================================================================

    # ==========================================================================================================
    # SUB REGION CROP
    # =====================================================================
    # call backs for each shape of the sub region select
    def select_sub_region_callback_elastic(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(buffer_image_2, (x, y), 2, (255, 0, 0), 2)
            sub_region_contour.append([[x, y]])
            cv2.drawContours(buffer_image_2, np.array([sub_region_contour[-2:]]), -1, (255, 0, 0), 1)

    def select_sub_region_callback_circle(event, x, y, flags, param):
        global select, temp, state
        if event == cv2.EVENT_LBUTTONDOWN and not select:
            cv2.circle(buffer_image_2, (x, y), 1, (0, 0, 255), -1)
            temp.append((x, y))
            select = True
            state = buffer_image_2.copy()
        elif event == cv2.EVENT_LBUTTONUP and select:
            center = ((temp[0][0] + x) // 2, (temp[0][1] + y) // 2)
            diameter = math.sqrt((temp[0][0] - x) ** 2 + (temp[0][1] - y) ** 2)
            sub_region_circles.append([center, int((diameter / 2) + 0.5)])
            cv2.circle(buffer_image_2, center, int((diameter / 2) + 0.5), (0, 0, 255), 2)
            temp = []
            select = False
        elif select:
            np.copyto(buffer_image_2, state)
            center = ((temp[0][0] + x) // 2, (temp[0][1] + y) // 2)
            diameter = math.sqrt((temp[0][0] - x) ** 2 + (temp[0][1] - y) ** 2)
            cv2.circle(buffer_image_2, center, int((diameter / 2) + 0.5), (0, 0, 255), 2)

    def select_sub_region_callback_square(event, x, y, flags, param):
        global select, temp, state
        if event == cv2.EVENT_LBUTTONDOWN and not select:
            cv2.circle(buffer_image_2, (x, y), 1, (0, 255, 0), -1)
            temp.append((x, y))
            select = True
            state = buffer_image_2.copy()
        elif event == cv2.EVENT_LBUTTONUP and select:
            sub_region_squares.append([temp[0], (x, y)])
            cv2.rectangle(buffer_image_2, temp[0], (x, y), (0, 255, 0), 2)
            temp = []
            select = False
        elif select:
            np.copyto(buffer_image_2, state)
            cv2.rectangle(buffer_image_2, temp[0], (x, y), (0, 255, 0), 2)

    # ===========================================================================
    # buffer image for reset
    buffer_image_2 = buffer_image.copy()
    # dict of functions
    callbacks = {0: select_sub_region_callback_elastic, 1: select_sub_region_callback_circle,
                 2: select_sub_region_callback_square}
    callback_names = {0: 'elastic crop', 1: 'circle crop',
                      2: 'square crop'}

    callback_selector = 0

    while 1:
        instruction_image = np.zeros((30, buffer_image_2.shape[1], 3), np.uint8)
        text = f'[r]-reset [q]-quit [n]-next(elastic crop only) [m]-mode, mode: {callback_names[callback_selector]}'
        instruction_image = put_text_on_image(instruction_image, buffer_image_2.shape[1], text)
        window_name = "Sub Region Select"
        cv2.namedWindow(window_name)
        params = [buffer_image_2.shape]
        cv2.setMouseCallback(window_name, callbacks[callback_selector],
                             param=params)

        cv2.imshow(window_name, cv2.vconcat([buffer_image_2, instruction_image]))
        key = cv2.waitKey(1)

        if key == 13:
            print("selecting sub region/s done....")
            if sub_region_contour:
                sub_region_contours.append(sub_region_contour)
            cv2.destroyAllWindows()
            break
        elif key == ord('n') and callback_names[callback_selector] == 'elastic crop':
            print("selecting next sub region....")
            if sub_region_contour:
                sub_region_contours.append(sub_region_contour)
            sub_region_contour = []
        elif key == ord('m'):
            callback_selector += 1 if callback_selector < 2 else -2
            print(callback_names[callback_selector])
            if sub_region_contour:
                sub_region_contours.append(sub_region_contour)
            sub_region_contour = []
        elif key == ord('r'):
            print("selecting sub region reset....")
            buffer_image_2 = buffer_image.copy()
            sub_region_contour = []
            sub_region_contours = []
            sub_region_circles = []
            sub_region_squares = []

        elif key == ord('q'):
            print("selecting sub region quit....")
            cv2.destroyAllWindows()
            return -1

    mask = np.zeros(image.shape, np.uint8)
    #  if contours in elastic crop then crop
    if sub_region_contours:
        for contour_ in sub_region_contours:
            cv2.drawContours(mask, np.array([contour_]), -1, (255, 255, 255), -1)
    # if circles them crop
    if sub_region_circles:
        for circle in sub_region_circles:
            cv2.circle(mask, circle[0], circle[1], (255, 255, 255), -1)
    # if squares then crop
    if sub_region_squares:
        for square in sub_region_squares:
            cv2.rectangle(mask, square[0], square[1], (255, 255, 255), -1)

    buffer_image = cv2.bitwise_and(buffer_image, cv2.bitwise_not(mask))
    # =============================================================================================================================

    image = cv2.cvtColor(buffer_image, cv2.COLOR_BGR2GRAY)

    if not sub_region_select:
        positions = np.nonzero(image)
        top = positions[0].min()
        bottom = positions[0].max()
        left = positions[1].min()
        right = positions[1].max()
        image = image[top:bottom, left:right]

    return image


if __name__ == '__main__':
    root = Tk()
    root.withdraw()
    root.filename = filedialog.askopenfilename(title="Select file",
                                               filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    image = select_region_elastic(cv2.imread(root.filename))
    show(image, 'Final Selected Region')
