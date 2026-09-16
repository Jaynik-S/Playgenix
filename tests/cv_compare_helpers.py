def safe_divide(numerator, denominator):
    if denominator == 0:
        return 1.0
    return numerator / denominator


def _timestamp_index(data):
    index = {}
    for key in data.keys():
        try:
            ts = int(key)
        except (TypeError, ValueError):
            continue
        if ts not in index:
            index[ts] = key
    return index


def compare_timestamps(expected, actual):
    expected_index = _timestamp_index(expected)
    actual_index = _timestamp_index(actual)

    expected_set = set(expected_index.keys())
    actual_set = set(actual_index.keys())
    matched = expected_set & actual_set

    missing = [expected_index[ts] for ts in sorted(expected_set - actual_set)]
    extra = [actual_index[ts] for ts in sorted(actual_set - expected_set)]

    return {
        "accuracy": safe_divide(len(matched), len(expected_set)),
        "expected_count": len(expected_set),
        "actual_count": len(actual_set),
        "matched_count": len(matched),
        "missing_timestamps": missing,
        "extra_timestamps": extra,
        "expected_index": expected_index,
        "actual_index": actual_index,
    }


def compare_game_status(expected, actual):
    timestamp_report = compare_timestamps(expected, actual)
    expected_index = timestamp_report["expected_index"]
    actual_index = timestamp_report["actual_index"]

    shared = sorted(set(expected_index.keys()) & set(actual_index.keys()))

    team_expected_agents = 0
    team_matched_agents = 0
    enemy_expected_agents = 0
    enemy_matched_agents = 0

    team_expected_bools = 0
    team_matched_bools = 0
    enemy_expected_bools = 0
    enemy_matched_bools = 0

    agent_name_mismatches = []
    ult_status_mismatches = []

    for ts in shared:
        expected_ts_key = expected_index[ts]
        actual_ts_key = actual_index[ts]
        expected_entry = expected.get(expected_ts_key, {}) or {}
        actual_entry = actual.get(actual_ts_key, {}) or {}

        for side in ("team", "enemy"):
            expected_agents = expected_entry.get(side, {}) or {}
            actual_agents = actual_entry.get(side, {}) or {}

            expected_names = set(expected_agents.keys())
            actual_names = set(actual_agents.keys())
            matched_names = expected_names & actual_names

            if side == "team":
                team_expected_agents += len(expected_names)
                team_matched_agents += len(matched_names)
            else:
                enemy_expected_agents += len(expected_names)
                enemy_matched_agents += len(matched_names)

            missing_agents = sorted(expected_names - actual_names)
            extra_agents = sorted(actual_names - expected_names)
            if missing_agents or extra_agents:
                agent_name_mismatches.append(
                    {
                        "timestamp": expected_ts_key,
                        "side": side,
                        "missing_agents": missing_agents,
                        "extra_agents": extra_agents,
                    }
                )

            for agent in sorted(expected_names):
                expected_val = expected_agents.get(agent)
                actual_val = actual_agents.get(agent, "MISSING")
                is_match = actual_val == expected_val

                if side == "team":
                    team_expected_bools += 1
                    if is_match:
                        team_matched_bools += 1
                else:
                    enemy_expected_bools += 1
                    if is_match:
                        enemy_matched_bools += 1

                if not is_match:
                    ult_status_mismatches.append(
                        {
                            "timestamp": expected_ts_key,
                            "side": side,
                            "agent": agent,
                            "expected": expected_val,
                            "actual": actual_val,
                        }
                    )

    return {
        "timestamp_accuracy": timestamp_report["accuracy"],
        "team_agent_name_accuracy": safe_divide(
            team_matched_agents, team_expected_agents
        ),
        "enemy_agent_name_accuracy": safe_divide(
            enemy_matched_agents, enemy_expected_agents
        ),
        "team_ult_status_accuracy": safe_divide(
            team_matched_bools, team_expected_bools
        ),
        "enemy_ult_status_accuracy": safe_divide(
            enemy_matched_bools, enemy_expected_bools
        ),
        "timestamp_counts": {
            "expected": timestamp_report["expected_count"],
            "actual": timestamp_report["actual_count"],
            "matched": timestamp_report["matched_count"],
        },
        "team_agent_name_counts": {
            "expected": team_expected_agents,
            "matched": team_matched_agents,
        },
        "enemy_agent_name_counts": {
            "expected": enemy_expected_agents,
            "matched": enemy_matched_agents,
        },
        "team_ult_status_counts": {
            "expected": team_expected_bools,
            "matched": team_matched_bools,
        },
        "enemy_ult_status_counts": {
            "expected": enemy_expected_bools,
            "matched": enemy_matched_bools,
        },
        "missing_timestamps": timestamp_report["missing_timestamps"],
        "extra_timestamps": timestamp_report["extra_timestamps"],
        "agent_name_mismatches": agent_name_mismatches,
        "ult_status_mismatches": ult_status_mismatches,
    }


def compare_slot_matches(expected, actual):
    timestamp_report = compare_timestamps(expected, actual)
    expected_index = timestamp_report["expected_index"]
    actual_index = timestamp_report["actual_index"]

    shared = sorted(set(expected_index.keys()) & set(actual_index.keys()))

    ability_expected_names = 0
    ability_matched_names = 0
    ability_expected_counts = 0
    ability_matched_counts = 0

    ability_name_mismatches = []
    ability_count_mismatches = []

    for ts in shared:
        expected_ts_key = expected_index[ts]
        actual_ts_key = actual_index[ts]
        expected_entry = expected.get(expected_ts_key, {}) or {}
        actual_entry = actual.get(actual_ts_key, {}) or {}

        expected_names = set(expected_entry.keys())
        actual_names = set(actual_entry.keys())
        matched_names = expected_names & actual_names

        ability_expected_names += len(expected_names)
        ability_matched_names += len(matched_names)

        missing_abilities = sorted(expected_names - actual_names)
        extra_abilities = sorted(actual_names - expected_names)
        if missing_abilities or extra_abilities:
            ability_name_mismatches.append(
                {
                    "timestamp": expected_ts_key,
                    "missing_abilities": missing_abilities,
                    "extra_abilities": extra_abilities,
                }
            )

        for ability in sorted(expected_names):
            expected_val = expected_entry.get(ability)
            actual_val = actual_entry.get(ability, "MISSING")

            ability_expected_counts += 1
            if actual_val == expected_val:
                ability_matched_counts += 1
            else:
                ability_count_mismatches.append(
                    {
                        "timestamp": expected_ts_key,
                        "ability": ability,
                        "expected": expected_val,
                        "actual": actual_val,
                    }
                )

    return {
        "timestamp_accuracy": timestamp_report["accuracy"],
        "ability_name_accuracy": safe_divide(
            ability_matched_names, ability_expected_names
        ),
        "ability_count_accuracy": safe_divide(
            ability_matched_counts, ability_expected_counts
        ),
        "timestamp_counts": {
            "expected": timestamp_report["expected_count"],
            "actual": timestamp_report["actual_count"],
            "matched": timestamp_report["matched_count"],
        },
        "ability_name_counts": {
            "expected": ability_expected_names,
            "matched": ability_matched_names,
        },
        "ability_count_counts": {
            "expected": ability_expected_counts,
            "matched": ability_matched_counts,
        },
        "missing_timestamps": timestamp_report["missing_timestamps"],
        "extra_timestamps": timestamp_report["extra_timestamps"],
        "ability_name_mismatches": ability_name_mismatches,
        "ability_count_mismatches": ability_count_mismatches,
    }
