"""Output formatting, logging, and JSON persistence helpers."""
import cv2
import json
import os


def log_error(message):
    print(message)


def log_video_resolution(frame_width, frame_height):
    print(f"\n\nVideo resolution: {frame_width}x{frame_height}")


def log_start_processing():
    print("STARTING FRAME PROCESSING")


def log_timestamp(timestamp_str):
    print(f"\n{timestamp_str}\n")


def log_ability_icon_match(ability_icon):
    print(f"Ability icon matched: {ability_icon}")


def log_charge_icon_match(charge_icon):
    print(f"Ability icon matched: {charge_icon}")


def log_agent_match(best_match, best_score, blank_slot=False):
    print(f"=====Best match: {best_match} with score {best_score:.4f}=====")
    if blank_slot:
        print("=====Slot is blank=====")


def log_ultimate_ratio(is_ready, ratio):
    print(f"{is_ready}: Ratio: {ratio:.2f}")


def log_agent_detected(agent, ult_ready):
    print(f"Agent detected: {agent}, Ultimate ready: {ult_ready}")


def log_processing_summary(ability_name, combined_status, team_status_dict, enemy_status_dict, slot_matches):
    print(f"Ability match: {ability_name}")
    print("=== COMBINED STATUS ===")
    for timestamp, status in combined_status.items():
        print(f"{timestamp}:")
        print(f"  Team : {status['team']}\n  Enemy: {status['enemy']}")

    print("=== TEAM STATUS ===")
    formatted_agents(team_status_dict)
    print("=== ENEMY STATUS ===")
    formatted_agents(enemy_status_dict)

    print("=== SLOT MATCHES ===")
    formatted_ability(slot_matches)


def log_json_saving():
    print("=== JSON SAVING ===")


def formatted_agents(data):
    for key, values in data.items():
        print(key)


def formatted_ability(data):
    for key, values in data.items():
        print(key)
        for value in values:
            print(value[1])


def show_video_frames(frames):
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    for frame in frames:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(0):
            continue
    cv2.destroyWindow("Frame")


def save_to_json(ability, status, slot_matches, file_name):
    agent_path = f"assets/json_data/ability_to_agent.json"
    with open(agent_path, "r") as f:
        data = json.load(f)
    agent = data[ability]

    ability_path = f"assets/json_data/agent_to_ability.json"
    with open(ability_path, "r") as f:
        data = json.load(f)
    agent_abilities = data[agent]

    converted = {}
    for timestamp, entries in slot_matches.items():
        converted[timestamp] = {agent_abilities[slot_key]: charge for slot_key, charge in entries}

    session_dir = os.path.join("app", "session_data")
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    print(f"\n\n\nBASENAME:, {base_name}\n\n\n")

    output_path = os.path.join(session_dir, f"{base_name}_slot_matches.json")
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=4)
    print(f"Slot matches saved to {output_path}")

    output_path = os.path.join(session_dir, f"{base_name}_game_status.json")
    with open(output_path, "w") as f:
        json.dump(_clean_null_agents(status), f, indent=4)
    print(f"Status saved to {output_path}")


def _clean_null_agents(status):
    cleaned = {}
    for timestamp, snapshot in status.items():
        team = snapshot.get("team", {})
        enemy = snapshot.get("enemy", {})
        cleaned[timestamp] = {
            "team": {k: v for k, v in team.items() if k not in (None, "null")},
            "enemy": {k: v for k, v in enemy.items() if k not in (None, "null")},
        }
    return cleaned
