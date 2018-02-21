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

    def average_slope_intercept(self, lines):
        left_lines = []  # (slope, intercept)
        left_weights = []  # (length,)
        right_lines = []  # (slope, intercept)
        right_weights = []  # (length,)

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1 or y1 == y2:
                    continue  # ignore a vertical, horizontal lines
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - (slope * x1)
                length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                if slope < 0:  # y is reversed in image
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))

        # add more weight to longer lines
        left_lane = np.dot(left_weights,  left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

        return left_lane, right_lane  # (slope, intercept), (slope, intercept)

    def make_line_points(self, y1, y2, line):
        """
        Convert a line represented in slope and intercept into pixel points
        """
        if line is None:
            return None

        slope, intercept = line

        # make sure everything is integer as cv2.line requires it
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        y1 = int(y1)
        y2 = int(y2)

        return ((x1, y1), (x2, y2))

    def lane_lines(self, image, lines):
        left_lane, right_lane = self.average_slope_intercept(lines)

        y1 = image.shape[0]  # bottom of the image
        y2 = y1 * 0.6         # slightly lower than the middle

        left_line = self.make_line_points(y1, y2, left_lane)
        right_line = self.make_line_points(y1, y2, right_lane)

        return left_line, right_line

    def draw_lane_lines(self, image, lines, color=[0, 0, 255], thickness=20):
        # make a separate image to draw lines and combine with the orignal later
        line_image = np.zeros_like(image)
        for line in lines:
            if line is not None:
                cv2.line(line_image, *line,  color, thickness)
        # image1 * α + image2 * β + λ
        # image1 and image2 must be the same shape.
        return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)

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
            points = np.array([[width * 0.05, height * 0.95],
                               [width * 0.40, height * 0.60],
                               [width * 0.60, height * 0.60],
                               [width * 0.95, height * 0.95]], dtype=np.int32)
            cv2.fillPoly(region, [points], (255, 255, 255))

            blur = cv2.GaussianBlur(frame, (5, 5), 0)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 250, 255)
            edges = cv2.bitwise_and(edges, edges, mask=region)

            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=40, minLineLength=20, maxLineGap=300)

            elapsedTime = cv2.getTickCount() - startTime
            fps = cv2.getTickFrequency() / elapsedTime

            if lines is not None:
                self.vis_in = self.draw_lane_lines(self.vis_in, self.lane_lines(self.vis_in, lines))

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
