import json
from pathlib import Path

import pytest

from cv_compare import compare_game_status, compare_slot_matches, safe_divide

TIMESTAMP_MIN = 1.00
AGENT_NAME_MIN = 1.00
ULT_STATUS_MIN = 1.00
ABILITY_NAME_MIN = 1.00
ABILITY_COUNT_MIN = 1.00

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DIR = REPO_ROOT / "tests" / "expected_outputs"
ACTUAL_DIR = REPO_ROOT / "app" / "session_data"
ARTIFACT_DIR = REPO_ROOT / "tests" / "artifacts"


def _list_video_ids(expected_dir, suffix):
    return sorted(
        path.name[: -len(suffix)]
        for path in expected_dir.glob(f"*{suffix}")
        if path.is_file()
    )


VIDEO_IDS_GAME_STATUS = _list_video_ids(EXPECTED_DIR, "_game_status.json")
VIDEO_IDS_SLOT_MATCHES = _list_video_ids(EXPECTED_DIR, "_slot_matches.json")


def _load_json(path):
    if not path.exists():
        pytest.fail(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_artifact(path, payload):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _format_pct(value):
    return f"{value * 100:.0f}%"


def _count_game_status_mismatches(report):
    return (
        len(report["missing_timestamps"])
        + len(report["extra_timestamps"])
        + len(report["agent_name_mismatches"])
        + len(report["ult_status_mismatches"])
    )


def _count_slot_match_mismatches(report):
    return (
        len(report["missing_timestamps"])
        + len(report["extra_timestamps"])
        + len(report["ability_name_mismatches"])
        + len(report["ability_count_mismatches"])
    )


def _compute_game_status_report(video_id):
    expected_path = EXPECTED_DIR / f"{video_id}_game_status.json"
    actual_path = ACTUAL_DIR / f"{video_id}_game_status.json"

    expected = _load_json(expected_path)
    actual = _load_json(actual_path)

    report = compare_game_status(expected, actual)
    report["video_id"] = video_id
    report["file"] = "game_status.json"
    report["mismatch_count"] = _count_game_status_mismatches(report)
    return report


def _compute_slot_match_report(video_id):
    expected_path = EXPECTED_DIR / f"{video_id}_slot_matches.json"
    actual_path = ACTUAL_DIR / f"{video_id}_slot_matches.json"

    expected = _load_json(expected_path)
    actual = _load_json(actual_path)

    report = compare_slot_matches(expected, actual)
    report["video_id"] = video_id
    report["file"] = "slot_matches.json"
    report["mismatch_count"] = _count_slot_match_mismatches(report)
    return report


def _compute_overall_summary(game_ids, slot_ids):
    game_reports = [_compute_game_status_report(v) for v in game_ids]
    slot_reports = [_compute_slot_match_report(v) for v in slot_ids]

    game_timestamp_expected = sum(
        r["timestamp_counts"]["expected"] for r in game_reports
    )
    game_timestamp_matched = sum(r["timestamp_counts"]["matched"] for r in game_reports)
    team_agent_expected = sum(
        r["team_agent_name_counts"]["expected"] for r in game_reports
    )
    team_agent_matched = sum(
        r["team_agent_name_counts"]["matched"] for r in game_reports
    )
    enemy_agent_expected = sum(
        r["enemy_agent_name_counts"]["expected"] for r in game_reports
    )
    enemy_agent_matched = sum(
        r["enemy_agent_name_counts"]["matched"] for r in game_reports
    )
    team_ult_expected = sum(
        r["team_ult_status_counts"]["expected"] for r in game_reports
    )
    team_ult_matched = sum(r["team_ult_status_counts"]["matched"] for r in game_reports)
    enemy_ult_expected = sum(
        r["enemy_ult_status_counts"]["expected"] for r in game_reports
    )
    enemy_ult_matched = sum(
        r["enemy_ult_status_counts"]["matched"] for r in game_reports
    )

    slot_timestamp_expected = sum(
        r["timestamp_counts"]["expected"] for r in slot_reports
    )
    slot_timestamp_matched = sum(r["timestamp_counts"]["matched"] for r in slot_reports)
    ability_name_expected = sum(
        r["ability_name_counts"]["expected"] for r in slot_reports
    )
    ability_name_matched = sum(
        r["ability_name_counts"]["matched"] for r in slot_reports
    )
    ability_count_expected = sum(
        r["ability_count_counts"]["expected"] for r in slot_reports
    )
    ability_count_matched = sum(
        r["ability_count_counts"]["matched"] for r in slot_reports
    )

    return {
        "game_status": {
            "timestamp_accuracy": safe_divide(
                game_timestamp_matched, game_timestamp_expected
            ),
            "team_agent_name_accuracy": safe_divide(
                team_agent_matched, team_agent_expected
            ),
            "team_ult_status_accuracy": safe_divide(
                team_ult_matched, team_ult_expected
            ),
            "enemy_agent_name_accuracy": safe_divide(
                enemy_agent_matched, enemy_agent_expected
            ),
            "enemy_ult_status_accuracy": safe_divide(
                enemy_ult_matched, enemy_ult_expected
            ),
        },
        "slot_matches": {
            "timestamp_accuracy": safe_divide(
                slot_timestamp_matched, slot_timestamp_expected
            ),
            "ability_name_accuracy": safe_divide(
                ability_name_matched, ability_name_expected
            ),
            "ability_count_accuracy": safe_divide(
                ability_count_matched, ability_count_expected
            ),
        },
    }


def _print_game_status_report(report):
    print(f"Video: {report['video_id']}")
    print("File: game_status.json")
    print(f"timestamp_accuracy: {_format_pct(report['timestamp_accuracy'])}")
    print(
        f"team_agent_name_accuracy: {_format_pct(report['team_agent_name_accuracy'])}"
    )
    print(
        f"team_ult_status_accuracy: {_format_pct(report['team_ult_status_accuracy'])}"
    )
    print(
        f"enemy_agent_name_accuracy: {_format_pct(report['enemy_agent_name_accuracy'])}"
    )
    print(
        f"enemy_ult_status_accuracy: {_format_pct(report['enemy_ult_status_accuracy'])}"
    )
    print(f"mismatches: {report['mismatch_count']}")


def _print_slot_match_report(report):
    print(f"Video: {report['video_id']}")
    print("File: slot_matches.json")
    print(f"timestamp_accuracy: {_format_pct(report['timestamp_accuracy'])}")
    print(f"ability_name_accuracy: {_format_pct(report['ability_name_accuracy'])}")
    print(f"ability_count_accuracy: {_format_pct(report['ability_count_accuracy'])}")
    print(f"mismatches: {report['mismatch_count']}")


@pytest.mark.parametrize("video_id", VIDEO_IDS_GAME_STATUS)
def test_game_status_accuracy(video_id):
    report = _compute_game_status_report(video_id)
    _print_game_status_report(report)
    _write_artifact(
        ARTIFACT_DIR / f"{video_id}_game_status_report.json",
        report,
    )

    assert report["timestamp_accuracy"] >= TIMESTAMP_MIN
    assert report["team_agent_name_accuracy"] >= AGENT_NAME_MIN
    assert report["enemy_agent_name_accuracy"] >= AGENT_NAME_MIN
    assert report["team_ult_status_accuracy"] >= ULT_STATUS_MIN
    assert report["enemy_ult_status_accuracy"] >= ULT_STATUS_MIN


@pytest.mark.parametrize("video_id", VIDEO_IDS_SLOT_MATCHES)
def test_slot_matches_accuracy(video_id):
    report = _compute_slot_match_report(video_id)
    _print_slot_match_report(report)
    _write_artifact(
        ARTIFACT_DIR / f"{video_id}_slot_matches_report.json",
        report,
    )

    assert report["timestamp_accuracy"] >= TIMESTAMP_MIN
    assert report["ability_name_accuracy"] >= ABILITY_NAME_MIN
    assert report["ability_count_accuracy"] >= ABILITY_COUNT_MIN


def test_overall_summary():
    summary = _compute_overall_summary(
        VIDEO_IDS_GAME_STATUS,
        VIDEO_IDS_SLOT_MATCHES,
    )

    print("Overall Summary")
    print("File: game_status.json")
    print(
        "timestamp_accuracy: "
        f"{_format_pct(summary['game_status']['timestamp_accuracy'])}"
    )
    print(
        "team_agent_name_accuracy: "
        f"{_format_pct(summary['game_status']['team_agent_name_accuracy'])}"
    )
    print(
        "team_ult_status_accuracy: "
        f"{_format_pct(summary['game_status']['team_ult_status_accuracy'])}"
    )
    print(
        "enemy_agent_name_accuracy: "
        f"{_format_pct(summary['game_status']['enemy_agent_name_accuracy'])}"
    )
    print(
        "enemy_ult_status_accuracy: "
        f"{_format_pct(summary['game_status']['enemy_ult_status_accuracy'])}"
    )

    print("File: slot_matches.json")
    print(
        "timestamp_accuracy: "
        f"{_format_pct(summary['slot_matches']['timestamp_accuracy'])}"
    )
    print(
        "ability_name_accuracy: "
        f"{_format_pct(summary['slot_matches']['ability_name_accuracy'])}"
    )
    print(
        "ability_count_accuracy: "
        f"{_format_pct(summary['slot_matches']['ability_count_accuracy'])}"
    )


def _print_direct_game_status(report):
    print("Game Status:")
    print(f"timestamp: {_format_pct(report['timestamp_accuracy'])}")
    print("team_agent_name: " f"{_format_pct(report['team_agent_name_accuracy'])}")
    print("team_ult_status: " f"{_format_pct(report['team_ult_status_accuracy'])}")
    print("enemy_agent_name: " f"{_format_pct(report['enemy_agent_name_accuracy'])}")
    print("enemy_ult_status: " f"{_format_pct(report['enemy_ult_status_accuracy'])}")


def _print_direct_slot_matches(report):
    print("Slot Matches:")
    print(f"timestamp: {_format_pct(report['timestamp_accuracy'])}")
    print(f"ability_name: {_format_pct(report['ability_name_accuracy'])}")
    print(f"ability_count: {_format_pct(report['ability_count_accuracy'])}")


def _print_direct_overall(summary):
    print("#######")
    print("OVERALL")
    print("#######")
    print(
        "game_status.timestamp_accuracy: "
        f"{_format_pct(summary['game_status']['timestamp_accuracy'])}"
    )
    print(
        "game_status.team_agent_name_accuracy: "
        f"{_format_pct(summary['game_status']['team_agent_name_accuracy'])}"
    )
    print(
        "game_status.team_ult_status_accuracy: "
        f"{_format_pct(summary['game_status']['team_ult_status_accuracy'])}"
    )
    print(
        "game_status.enemy_agent_name_accuracy: "
        f"{_format_pct(summary['game_status']['enemy_agent_name_accuracy'])}"
    )
    print(
        "game_status.enemy_ult_status_accuracy: "
        f"{_format_pct(summary['game_status']['enemy_ult_status_accuracy'])}"
    )
    print(
        "slot_matches.timestamp_accuracy: "
        f"{_format_pct(summary['slot_matches']['timestamp_accuracy'])}"
    )
    print(
        "slot_matches.ability_name_accuracy: "
        f"{_format_pct(summary['slot_matches']['ability_name_accuracy'])}"
    )
    print(
        "slot_matches.ability_count_accuracy: "
        f"{_format_pct(summary['slot_matches']['ability_count_accuracy'])}"
    )


if __name__ == "__main__":
    requested_ids = []
    if requested_ids:
        video_ids = requested_ids
    else:
        video_ids = sorted(set(VIDEO_IDS_GAME_STATUS + VIDEO_IDS_SLOT_MATCHES))

    for video_id in video_ids:
        has_game = (EXPECTED_DIR / f"{video_id}_game_status.json").exists()
        has_slot = (EXPECTED_DIR / f"{video_id}_slot_matches.json").exists()
        if not (has_game or has_slot):
            continue

        print("#######")
        print(video_id)
        print("#######")

        if has_game:
            game_report = _compute_game_status_report(video_id)
            _print_direct_game_status(game_report)
        if has_slot:
            slot_report = _compute_slot_match_report(video_id)
            _print_direct_slot_matches(slot_report)

    game_ids = [
        video_id
        for video_id in video_ids
        if (EXPECTED_DIR / f"{video_id}_game_status.json").exists()
    ]
    slot_ids = [
        video_id
        for video_id in video_ids
        if (EXPECTED_DIR / f"{video_id}_slot_matches.json").exists()
    ]
    if game_ids or slot_ids:
        summary = _compute_overall_summary(game_ids, slot_ids)
        _print_direct_overall(summary)
