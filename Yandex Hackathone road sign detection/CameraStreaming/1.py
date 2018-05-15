import numpy as np
import cv2 as cv
import cv2

file = ''
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        #Отображение текста на кадрах видеопотока (вставьте наименование знака вместо helloworld)
        ## TODO: Напишите код для детектирования и распознавания знака
        import numpy as np

        names = {1: ["Treangle", np.array([119, 101, 73]), np.array([255, 255, 255]), 75, np.array([0, 0, 0]),
                     np.array([255, 255, 254])],  # !!!! -
                 2: ["Doing works", np.array([20, 71, 71]), np.array([56, 255, 255]), 70, np.array([30, 30, 255]),
                     np.array([255, 255, 255])],
                 3: ["Crosswalk", np.array([98, 57, 60]), np.array([163, 255, 255]), 87,
                     np.array([25, 82, 117]), np.array([255, 255, 253])],  # !!!!
                 4: ["Stand_place", np.array([98, 57, 60]), np.array([163, 255, 255]), 65, np.array([16, 82, 17]),
                     np.array([233, 230, 254])],
                 5: ["Brick", np.array([119, 101, 73]), np.array([255, 255, 255]), 70, np.array([0, 0, 0]),
                     np.array([255, 255, 254])],
                 6: ["Stop", np.array([119, 101, 73]), np.array([255, 255, 255]), 70, np.array([0, 0, 0]),
                     np.array([255, 255, 254])],
                 7: ["Warning", np.array([119, 101, 73]), np.array([255, 255, 255]), 70, np.array([0, 0, 0]),
                     np.array([255, 255, 254])],
                 8: ["Tire", np.array([119, 101, 73]), np.array([255, 255, 255]), 70, np.array([0, 0, 0]),
                     np.array([255, 255, 254])], }
        # 9: ["Главная дорога", np.array([19, 24, 89]), np.array([35, 255, 255]),       80, np.array([30, 0, 151]), np.array([255, 255, 255])]}
        #
        images = {}
        for i in range(1, len(names)):
            image = cv.imread('Знаки/' + str(i) + '.png')
            image = cv.resize(image, (100, 100))
            images[names[i][0]] = image

        while (True):
            text = ''
            ret, frame = cap.read()
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            hsv = cv.blur(hsv, (5, 5))  # Размытие

            # Фильтр
            for image in range(1, len(names)):
                lower = names[image][1]
                upper = names[image][2]

                thresh = cv.inRange(hsv, lower, upper)
                thresh = cv.erode(thresh, None, iterations=2)
                thresh = cv.dilate(thresh, None, iterations=10)

                # Контуры
                contours = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                contours = contours[1]
                try:
                    c = sorted(contours, key=cv.contourArea, reverse=True)[0]
                    rect = cv.minAreaRect(c)
                    box = np.int0(cv.boxPoints(rect))
                    y1 = int(box[0][1])
                    x2 = int(box[1][0])
                    y2 = int(box[1][1])
                    x3 = int(box[2][0])

                    roiImg = frame[y2:y1, x2:x3]

                    if roiImg.any():
                        resizedRoi = cv.resize(roiImg, (100, 100))
                        # noDrive=cv.resize(noDrive, (100, 100))

                        xresizedRoi = cv.inRange(resizedRoi, lower, upper)
                        xnoDrive = cv.inRange(images[names[image][0]], names[image][4], names[image][5])

                        identity_percent = 0
                        for i in range(100):
                            for j in range(100):
                                if (xresizedRoi[i][j] == xnoDrive[i][j]):
                                    identity_percent += 1
                        # print(identity_percent/100)
                        identity_percent /= 100
                        if identity_percent > names[image][3]:
                            cv.drawContours(frame, [box], -1, (0, 255, 0), 3)  # draw contours in green color
                            cv2.putText(frame, names[image][0], (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


                except:
                    pass

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # Пример отображения текста на видео

        # Отображение видео (необходимо убрать при запуске по ssh):
        #cv2.imshow('frame', frame)
        out.write(frame)


    else:
        break

cap.release()
out.release()

cv2.destroyAllWindows()
