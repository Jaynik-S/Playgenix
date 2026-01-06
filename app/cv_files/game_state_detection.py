"""Agent and ultimate detection helpers."""
import cv2
import os
import numpy as np


def match_agent_slot(
    screenshot,
    bbox,
    agent_folder,
    visualize=False,
    threshold=0.15,
):
    x, y, w, h = bbox
    cropped = screenshot[y:y+h, x:x+w]

    if visualize:
        cv2.imshow("Screenshot", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    best_score = float("-inf")
    best_match = None

    for agent_name in os.listdir(agent_folder):
        agent_path = os.path.join(agent_folder, agent_name)
        icon = cv2.imread(agent_path)

        h, w = cropped.shape[:2]
        if icon.shape[:2] != (h, w):
            icon = cv2.resize(icon, (w, h))

        res = cv2.matchTemplate(cropped, icon, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res)

        if score > best_score:
            best_score = score
            best_match = agent_name

    blank_slot = is_blank_slot_edges(cropped)
    return best_match, best_score, blank_slot


def detect_agent(
    screenshot,
    bbox,
    ult_bbox,
    agent_folder,
    visualize=False,
    threshold=0.15,
    return_debug=False,
):
    """Detect which agent (if any) occupies a slot and whether their ultimate is ready."""
    best_match, best_score, blank_slot = match_agent_slot(
        screenshot,
        bbox,
        agent_folder,
        visualize=visualize,
        threshold=threshold,
    )

    if blank_slot:
        agent = None
        ult_ready = None
        ultimate_ratio = None
    else:
        agent = os.path.splitext(best_match)[0]
        ult_ready, ultimate_ratio = detect_ultimate(
            screenshot,
            ult_bbox,
            visualize,
            return_ratio=True,
        )

    if return_debug:
        return (
            agent,
            ult_ready,
            {
                "best_match": best_match,
                "best_score": best_score,
                "blank": blank_slot,
                "ultimate_ready": ult_ready,
                "ultimate_ratio": ultimate_ratio,
            },
        )

    return agent, ult_ready


def is_blank_slot_edges(cropped_image, edge_threshold=0.05):
    """Check if slot is blank using edge detection."""
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    total_pixels = edges.shape[0] * edges.shape[1]
    edge_pixels = cv2.countNonZero(edges)
    edge_ratio = edge_pixels / total_pixels

    return edge_ratio < edge_threshold


def detect_ultimate(screenshot, bbox, visualize=False, threshold=0.1, return_ratio=False):
    x, y, w, h = bbox
    cropped = screenshot[y:y+h, x:x+w]

    if visualize:
        cv2.imshow("Ultimate Screenshot", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([25, 40, 140])
    upper_yellow = np.array([100, 120, 220])

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    yellow_pixels = cv2.countNonZero(yellow_mask)
    total_pixels = cropped.shape[0] * cropped.shape[1]
    yellow_ratio = yellow_pixels / float(total_pixels)

    if visualize:
        cv2.imshow("Cropped Ultimate Slot", cropped)
        cv2.imshow("Yellow Mask", yellow_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    is_ready = yellow_ratio > threshold
    if return_ratio:
        return is_ready, yellow_ratio
    return is_ready
