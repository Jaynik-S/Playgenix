"""Pipeline entry point that orchestrates the CV flow."""
import cv2
import os
from collections import defaultdict

from .ability_charge_detection import match_icon, preprocess_charge_crop
from .game_state_detection import detect_ultimate, match_agent_slot
from model_setup.img_preprocess import match_charge_cnn
from .output_utils import (
    log_agent_detected,
    log_agent_match,
    log_ability_icon_match,
    log_charge_icon_match,
    log_error,
    log_json_saving,
    log_processing_summary,
    log_start_processing,
    log_timestamp,
    log_ultimate_ratio,
    log_video_resolution,
    save_to_json,
    show_video_frames,
)
from .temporal_analysis import extract_frames, initialize_timestamp, spike_check


# GLOBAL
BASE_W, BASE_H = 1920, 1080
BASE_BOXES = {
    "ability_bbox": (759, 974, 60, 60),
    "charge_bboxes": [(740, 1036, 100, 23), (852, 1036, 100, 23), (965, 1036, 100, 23), (1078, 1036, 100, 23)],
    "clock_bbox": (930, 27, 65, 44),
    "spike_bbox": (918, 11, 85, 85),
    "team_status": [(444, 27, 45, 44), (509, 27, 45, 44), (573, 27, 45, 44), (641, 27, 45, 44), (708, 27, 45, 44)],
    "team_ultimate": [(444, 16, 45, 14), (509, 16, 45, 14), (573, 16, 45, 14), (641, 16, 45, 14), (708, 16, 45, 14)],
    "enemy_status": [(1169, 27, 45, 44), (1235, 27, 45, 44), (1301, 27, 45, 44), (1368, 27, 45, 44), (1434, 27, 45, 44)],
    "enemy_ultimate": [(1169, 16, 45, 14), (1235, 16, 45, 14), (1301, 16, 45, 14), (1368, 16, 45, 14), (1434, 16, 45, 14)],
}


def _scale_box(box, sx, sy):
    x, y, w, h = box
    return (int(round(x * sx)), int(round(y * sy)), int(round(w * sx)), int(round(h * sy)))


def scale_boxes(frame_w, frame_h):
    sx, sy = frame_w / BASE_W, frame_h / BASE_H
    out = {}
    for k, v in BASE_BOXES.items():
        if isinstance(v, list):
            out[k] = [_scale_box(b, sx, sy) for b in v]
        else:
            out[k] = _scale_box(v, sx, sy)
    return out


def main(file_name: str, visualize: bool = False, debug_dump_dir=None, use_edge_matching=True):
    #video_path = f"app/static/uploads/{file_name}"
    video_path = file_name

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, screenshot = cap.read()
    cap.release()
    if not ret:
        log_error("Error: Could not read frame from video")
        exit()

    log_video_resolution(frame_width, frame_height)
    boxes = scale_boxes(frame_width, frame_height)
    ability_bbox = boxes["ability_bbox"]
    charge_bboxes = boxes["charge_bboxes"]
    clock_bbox = boxes["clock_bbox"]
    spike_bbox = boxes["spike_bbox"]
    team_status = boxes["team_status"]
    team_ultimate = boxes["team_ultimate"]
    enemy_status = boxes["enemy_status"]
    enemy_ultimate = boxes["enemy_ultimate"]

    ability_matches = defaultdict(int)
    slot_matches = defaultdict(list)
    team_status_dict = defaultdict(dict)
    enemy_status_dict = defaultdict(dict)

    frames = extract_frames(video_path, interval=1)

    if visualize:
        show_video_frames(frames)

    spike_planted = spike_check(frames[0], spike_bbox, visualize)
    timestamp_str = initialize_timestamp(visualize, clock_bbox, frames, spike_planted)
    log_start_processing()
    for frame_idx, frame in enumerate(frames):
        timestamp_str -= 1
        log_timestamp(timestamp_str)

        if not spike_planted:
            spike_planted = spike_check(frame, spike_bbox, visualize)
            if spike_planted:
                timestamp_str = -1

        if max(ability_matches.values(), default=0) < 3:
            ability_icon = match_icon(
                frame,
                "assets/abilities",
                ability_bbox,
                visualize,
                use_edges=use_edge_matching,
                debug_dump_dir=debug_dump_dir,
                debug_tag=f"{timestamp_str}_f{frame_idx}",
            )
            log_ability_icon_match(ability_icon)
            ability_matches[ability_icon] += 1

        for slot_idx in range(len(charge_bboxes)):
            x, y, w, h = charge_bboxes[slot_idx]
            cropped = frame[y:y+h, x:x+w]

            target_size = (128, 32)
            resized_crop = preprocess_charge_crop(
                cropped,
                target_size=target_size,
                debug_dump_dir=debug_dump_dir,
                debug_tag=f"{timestamp_str}_f{frame_idx}_slot{slot_idx + 1}",
            )

            if visualize:
                cv2.imshow("Resized Crop", resized_crop)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            charge_icon = match_charge_cnn(resized_crop, None, (0, 0, target_size[0], target_size[1]))
            slot_matches[timestamp_str].append((f"Ability #{slot_idx + 1}", charge_icon[0:3]))
            log_charge_icon_match(charge_icon[0:3])

        for slot_idx in range(len(team_status)):
            best_match, best_score, blank_slot = match_agent_slot(
                frame,
                team_status[slot_idx],
                "assets/agents/normal",
                visualize,
            )
            log_agent_match(best_match, best_score, blank_slot)
            if blank_slot:
                agent = None
                ult_ready = None
                ultimate_ratio = None
            else:
                agent = os.path.splitext(best_match)[0]
                ult_ready, ultimate_ratio = detect_ultimate(
                    frame,
                    team_ultimate[slot_idx],
                    visualize,
                    return_ratio=True,
                )
            if visualize and ultimate_ratio is not None:
                log_ultimate_ratio(ult_ready, ultimate_ratio)
            log_agent_detected(agent, ult_ready)
            team_status_dict[timestamp_str][agent] = bool(ult_ready)

            best_match, best_score, blank_slot = match_agent_slot(
                frame,
                enemy_status[slot_idx],
                "assets/agents/flipped",
                visualize,
            )
            log_agent_match(best_match, best_score, blank_slot)
            if blank_slot:
                agent = None
                ult_ready = None
                ultimate_ratio = None
            else:
                agent = os.path.splitext(best_match)[0]
                ult_ready, ultimate_ratio = detect_ultimate(
                    frame,
                    enemy_ultimate[slot_idx],
                    visualize,
                    return_ratio=True,
                )
            if visualize and ultimate_ratio is not None:
                log_ultimate_ratio(ult_ready, ultimate_ratio)
            log_agent_detected(agent, ult_ready)
            enemy_status_dict[timestamp_str][agent] = bool(ult_ready)

    combined_status = {}
    for timestamp in team_status_dict.keys():
        combined_status[timestamp] = {
            "team": team_status_dict[timestamp],
            "enemy": enemy_status_dict[timestamp],
        }
    max_key = max(ability_matches, key=ability_matches.get)

    ability_name = max_key[:len(max_key)-4]
    log_processing_summary(ability_name, combined_status, team_status_dict, enemy_status_dict, slot_matches)

    log_json_saving()
    save_to_json(ability_name, combined_status, slot_matches, file_name)


if __name__ == "__main__":
    video_paths = ["v720", "v720-2", "v1080", "v1440", "v1080-2", "v1440-2", "v1080-3", "v1440-3"]
    for video in video_paths:
        main(f"videos/{video}.mp4", False)
