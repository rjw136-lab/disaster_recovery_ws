#!/usr/bin/env python3
import argparse
import cv2

ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

def main():
    parser = argparse.ArgumentParser(description="Generate an ArUco Marker Image")
    parser.add_argument("--id", type=int, default=0, help="Marker ID (default: 0)")
    parser.add_argument("--dict", type=str, default="DICT_4X4_50", help="Dictionary name")
    parser.add_argument("--size", type=int, default=400, help="Image size in pixels (default: 400)")
    parser.add_argument("--output", type=str, default="marker.png", help="Output PNG path (default: marker.png)")

    args = parser.parse_args()

    if args.dict not in ARUCO_DICTS:
        print(f"Error: Unknown dictionary {args.dict}")
        return

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[args.dict])
    marker_img = cv2.aruco.drawMarker(dictionary, args.id, args.size)
    cv2.imwrite(args.output, marker_img)
    print(f"Marker saved to {args.output} (ID: {args.id}, Dict: {args.dict}, Size: {args.size}x{args.size})")

if __name__ == "__main__":
    main()
