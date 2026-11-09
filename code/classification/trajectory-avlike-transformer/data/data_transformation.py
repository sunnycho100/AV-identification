import pandas as pd
import numpy as np
from pathlib import Path

earth_radius = 6371008.8
time_step = 0.1
speed_mismatch_threshold = 2.0
min_lv_fv_projection = 0.0
min_trajectory_frames = 50


def _to_radians(values):
    values = values.astype(float)
    if values.abs().max() > 2 * np.pi:
        return np.radians(values)
    return values


def _add_processed_coordinates(data):
    data = data.copy()
    for vehicle_type in ["LV", "FV"]:
        data[f"processed_x_{vehicle_type}"] = np.nan
        data[f"processed_y_{vehicle_type}"] = np.nan
        data[f"processed_z_{vehicle_type}"] = np.nan

    for _, group in data.groupby("Trajectory_ID", sort=False):
        if group.empty:
            continue

        origin_lat = _to_radians(group["Lat_FV"]).iloc[0]
        origin_lon = _to_radians(group["Lon_FV"]).iloc[0]
        origin_alt = group["Alt_FV"].astype(float).iloc[0]

        for vehicle_type in ["LV", "FV"]:
            lat = _to_radians(group[f"Lat_{vehicle_type}"])
            lon = _to_radians(group[f"Lon_{vehicle_type}"])
            alt = group[f"Alt_{vehicle_type}"].astype(float)

            data.loc[group.index, f"processed_x_{vehicle_type}"] = (
                earth_radius * (lon - origin_lon) * np.cos(origin_lat)
            )
            data.loc[group.index, f"processed_y_{vehicle_type}"] = -earth_radius * (lat - origin_lat)
            data.loc[group.index, f"processed_z_{vehicle_type}"] = alt - origin_alt

    return data


def _add_processed_speed_components(data):
    data = data.copy()
    for vehicle_type in ["LV", "FV"]:
        data[f"Speed_x_{vehicle_type}"] = np.nan
        data[f"Speed_y_{vehicle_type}"] = np.nan

    for _, group in data.groupby("Trajectory_ID", sort=False):
        if group.empty:
            continue

        time = group["Time_Index"].to_numpy(dtype=float)
        for vehicle_type in ["LV", "FV"]:
            for axis in ["x", "y"]:
                position = group[f"processed_{axis}_{vehicle_type}"].to_numpy(dtype=float)
                speed = np.zeros(len(group))

                if len(group) > 1:
                    dt = np.diff(time)
                    dp = np.diff(position)
                    segment_speed = np.divide(
                        dp,
                        dt,
                        out=np.zeros_like(dp),
                        where=dt != 0,
                    )
                    speed[:-1] = segment_speed
                    speed[-1] = segment_speed[-1]

                data.loc[group.index, f"Speed_{axis}_{vehicle_type}"] = speed

    return data


def _filter_coordinate_speed_mismatch(data):
    keep_mask = pd.Series(True, index=data.index)

    for vehicle_type in ["LV", "FV"]:
        coordinate_speed = np.sqrt(
            data[f"Speed_x_{vehicle_type}"] ** 2 + data[f"Speed_y_{vehicle_type}"] ** 2
        )
        speed_diff = (coordinate_speed - data[f"Speed_{vehicle_type}"]).abs()
        keep_mask &= np.isfinite(coordinate_speed)
        keep_mask &= np.isfinite(data[f"Speed_{vehicle_type}"])
        keep_mask &= speed_diff <= speed_mismatch_threshold

    return data.loc[keep_mask].copy()


def _filter_lv_in_front_of_fv(data):
    relative_x = data["processed_x_LV"] - data["processed_x_FV"]
    relative_y = data["processed_y_LV"] - data["processed_y_FV"]
    fv_speed_magnitude = np.sqrt(data["Speed_x_FV"] ** 2 + data["Speed_y_FV"] ** 2)
    projection = np.divide(
        relative_x * data["Speed_x_FV"] + relative_y * data["Speed_y_FV"],
        fv_speed_magnitude,
        out=np.full(len(data), np.nan),
        where=fv_speed_magnitude != 0,
    )

    keep_mask = np.isfinite(projection) & (projection >= min_lv_fv_projection)
    return data.loc[keep_mask].copy()


def _split_and_filter_trajectories(data):
    trajectory_list = []
    next_trajectory_id = 0

    for _, group in data.groupby("Trajectory_ID", sort=False):
        group = group.sort_values("Time_Index").copy()
        if group.empty:
            continue

        time_diff = group["Time_Index"].diff().fillna(time_step)
        segment_ids = (time_diff > time_step + 1e-9).cumsum()

        for _, segment in group.groupby(segment_ids, sort=False):
            if len(segment) < min_trajectory_frames:
                continue

            segment = segment.copy()
            segment["Trajectory_ID"] = next_trajectory_id
            segment["Time_Index"] = (
                ((segment["Time_Index"] - segment["Time_Index"].iloc[0]) / time_step)
                .round()
                .astype(int)
                * time_step
            )
            trajectory_list.append(segment)
            next_trajectory_id += 1

    if not trajectory_list:
        return data.iloc[0:0].copy()

    return pd.concat(trajectory_list, ignore_index=True)


def _prepare_output_data(data, output_columns):
    output_data = data[output_columns].copy()
    output_data["Time_Index"] = output_data["Time_Index"].map(lambda value: f"{value:.1f}")
    return output_data


def OpenACC_convert_format(input_path, output_path, id_map):
    df = pd.read_csv(input_path, header=None, skiprows=1, nrows=1)
    id_row = df.values.tolist()[0][1:]
    id_row = [x for x in id_row if not pd.isna(x)]
    vehicle_ids = list(map(lambda x: id_map.get(x, x), id_row))
    vehicle_num = len(vehicle_ids)

    raw_data = pd.read_csv(input_path, skiprows=5)

    df_list = []
    for i in range(vehicle_num):
        columns = [
            "Time",
            f"Speed{i}",
            f"Speed{i + 1}",
            f"Lat{i}",
            f"Lon{i}",
            f"Alt{i}",
            f"Lat{i + 1}",
            f"Lon{i + 1}",
            f"Alt{i + 1}",
        ]
        if not all(column in raw_data.columns for column in columns):
            continue

        df = raw_data[columns].copy()
        df.columns = [
            "Time_Index",
            "Speed_LV",
            "Speed_FV",
            "Lat_LV",
            "Lon_LV",
            "Alt_LV",
            "Lat_FV",
            "Lon_FV",
            "Alt_FV",
        ]
        df.insert(1, "ID_LV", vehicle_ids[i - 1])
        df.insert(3, "ID_FV", vehicle_ids[i])
        df.insert(0, "Trajectory_ID", i - 1)
        df_list.append(df)

    output_columns = [
        "Trajectory_ID",
        "Time_Index",
        "ID_LV",
        "Speed_LV",
        "Speed_x_LV",
        "Speed_y_LV",
        "processed_x_LV",
        "processed_y_LV",
        "processed_z_LV",
        "ID_FV",
        "Speed_FV",
        "Speed_x_FV",
        "Speed_y_FV",
        "processed_x_FV",
        "processed_y_FV",
        "processed_z_FV",
    ]

    if not df_list:
        pd.DataFrame(columns=output_columns).to_csv(output_path, index=False)
        return

    data = pd.concat(df_list, ignore_index=True)

    time_min = data.groupby("Trajectory_ID")["Time_Index"].transform("min")
    data["Time_Index"] = (
        ((data["Time_Index"] - time_min) / time_step).round().astype(int) * time_step
    )

    data = _add_processed_coordinates(data)
    data = _add_processed_speed_components(data)
    data = _filter_coordinate_speed_mismatch(data)
    data = _filter_lv_in_front_of_fv(data)
    data = _split_and_filter_trajectories(data)
    data = _add_processed_coordinates(data)
    data = _add_processed_speed_components(data)
    data = _prepare_output_data(data, output_columns)

    data.to_csv(output_path, index=False)


def main():
    raw_data_dir = Path("raw data")
    output_dir = Path("converted data")
    output_dir.mkdir(exist_ok=True)

    id_map = {
        "Ford(S-Max)": 0,
        "KIA(Niro)": 1,
        "Mini(Cooper)": 2,
        "Mitsubishi(OutlanderPHEV)": 3,
        "Mitsubishi(SpaceStar)": 4,
        "Peugeot(3008GTLine)": 5,
        "VW(GolfE)": 6,
    }

    for input_path in sorted(raw_data_dir.glob("*.csv")):
        if "260219" in input_path.name:
            data_type = "AV"
        elif "280219" in input_path.name:
            data_type = "HV"
        else:
            continue

        output_path = output_dir / f"{data_type}_{input_path.name}"
        OpenACC_convert_format(input_path, output_path, id_map)
        print(f"Converted {input_path} -> {output_path}")


if __name__ == "__main__":
    main()

