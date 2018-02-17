import cv2
import os
import numpy as np
import argparse
import logging
import time
import random


class LaneDetect(object):
    """
    Class for Lane Detection
    """
    def __init__(self):
        self.vis_in = None

    def runTest(self, fileName):
        cap = cv2.VideoCapture(fileName)
        assert cap.isOpened(), "Failed to read %s" % (fileName)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frameIdx in range(totalFrames):
            ret, frame = cap.read()
            if not ret:
                break
            self.vis_in = np.copy(frame)
            startTime = cv2.getTickCount()

            region = np.zeros(self.vis_in.shape[:2], dtype=np.uint8)
            points = np.array([[0, height], [width / 2, height / 2], [width, height]], dtype=np.int32)
            cv2.fillPoly(region, [points], (255, 255, 255))
            cv2.imshow("region", region)

            blur = cv2.GaussianBlur(frame, (3, 3), 0)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            cv2.imshow("gray", gray)
            edges = cv2.Canny(gray, 50, 150)
            cv2.imshow("edges", edges)

            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=20)

            elapsedTime = cv2.getTickCount() - startTime
            fps = cv2.getTickFrequency() / elapsedTime

            for i in range(len(lines)):
                # color = (random.uniform(0, 255), random.uniform(0, 255), random.uniform(0, 255))
                color = (80, 80, 180)
                for x1, y1, x2, y2 in lines[i]:
                    cv2.line(self.vis_in, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            text = "Frame:%03d, fps:%02d" % (frameIdx, fps)
            textProp, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
            org = (0, textProp[1])
            background = np.zeros(self.vis_in.shape, dtype=np.uint8)
            cv2.rectangle(background, (0, 0), (textProp[0], textProp[1] + baseline), (120, 120, 120), -1)
            self.vis_in = cv2.addWeighted(self.vis_in, 0.8, background, 1, 0)
            cv2.putText(self.vis_in, text, org, cv2.FONT_HERSHEY_DUPLEX, 0.8, (10, 10, 10), 2, cv2.LINE_AA)

            cv2.imshow("vis_in", self.vis_in)
            key = cv2.waitKey(30)
            if key == 32:
                cv2.waitKey(0)
            if key == 27:
                break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logging.info("Command Options: \n --input: %s" % (args.input))

    assert os.path.exists(args.input), "%s does not exist" % (args.input)

    laneDetect = LaneDetect()
    laneDetect.runTest(args.input)
