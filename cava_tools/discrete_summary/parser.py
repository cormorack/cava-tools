import re
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from loguru import logger
from lxml import etree

from cava_tools.discrete_summary.validator import check_name


def get_ds_labels(cols: str) -> dict:
    """
    Parses discrete samples labels and turns them into a dictionary.
    It turns the name to lower case and split up the units.

    Parameters
    ----------
    cols: list
        A listing of all the labels to be parsed.

    Returns
    -------
    dict

    """
    names = []
    display_name = []
    units = []
    for col in cols:
        match = re.search(r"(((\w+-?\w?)\s?)+)(\[.*\])?", col)
        if match:
            matches = match.groups()
            name = matches[0].strip()
            unit = matches[-1]
            if unit:
                unit = unit.strip("[]")
            name = check_name(name)
            if "name" != "unnamed":
                names.append(name.lower().replace(" ", "_"))
                display_name.append(name)
                units.append(unit)

    # for later, maybe save into separate table?
    discrete_samples_labels = {
        "name": names,
        "display_name": display_name,
        "unit": units,
    }
    return discrete_samples_labels


def get_folder_contents(folder_url: str) -> pd.DataFrame:
    """
    Parses and retrieves alfresco folder content from url.

    Parameters
    ----------
    folder_url : str
        Alfresco folder url

    Returns
    -------
    pd.DataFrame
    """
    pr = urlparse(folder_url)
    req = requests.get(folder_url)
    if req.status_code == 200:
        html = etree.HTML(req.content)
        content_tables = [
            t
            for t in html.xpath("//table")
            if "class" in t.attrib and t.attrib["class"] == "recordSet"
        ]
        for c in content_tables[1].getchildren():
            headers = []
            all_files = {}
            last_fname = ""
            for i in c.iter():
                if i.text:
                    if "class" in i.attrib:
                        headers.append(i.text)
                    elif "target" in i.attrib and i.attrib["target"] == "new":
                        fdct = {
                            "name": i.text,
                            "url": f"{pr.scheme}://{pr.netloc}{i.attrib['href']}",  # noqa
                            "description": "",
                            "size": "",
                            "created": "",
                            "modified": "",
                        }
                        all_files[fdct["name"]] = fdct
                        last_fname = fdct["name"]
                    elif "id" in i.attrib:
                        if "col13-txt" in i.attrib["id"]:
                            all_files[last_fname]["description"] = i.text
                        elif "col15-txt" in i.attrib["id"]:
                            all_files[last_fname]["size"] = i.text
                        elif "col16-txt" in i.attrib["id"]:
                            all_files[last_fname]["created"] = i.text
                        elif "col17-txt" in i.attrib["id"]:
                            all_files[last_fname]["modified"] = i.text
        df = pd.DataFrame(list(all_files.values()))
        df.loc[:, "modified"] = df.modified.apply(pd.to_datetime)
        df.loc[:, "created"] = df.created.apply(pd.to_datetime)
        return df
    else:
        logger.warning(f"Error found. {req.status_code}")


def _convert_dt(time_str):
    try:
        dt = pd.to_datetime(time_str)
    except Exception:
        logger.warning(f"Invalid time str: {time_str}")
        dt = np.NaN
    return dt


def clean_discrete_summary(
    svdf: pd.DataFrame, expected_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Cleans discrete summary data

    Parameters
    ----------
    svdf : pd.DataFrame
        Discrete summary data
    expected_columns : list
        Optional expected column to check against

    Returns
    -------
    tuple
    """
    cleaned = svdf.dropna(how="all")
    unnamed_cols = [c for c in cleaned.columns if "unnamed" in c.lower()]
    if len(unnamed_cols) > 0:
        logger.warning("EXTRA UNNAMED Columns Found! Fixing ...")
        cleaned = cleaned.drop(unnamed_cols, axis=1)
    cols = cleaned.columns

    # for later, maybe save into separate table?
    discrete_samples_labels = get_ds_labels(cols)
    names = discrete_samples_labels["name"]
    if expected_columns:
        missing_cols = [c for c in expected_columns if c not in names]
        if len(missing_cols) > 0:
            logger.warning(f"MISSING COLUMNS: {', '.join(missing_cols)}")

    cleaned.columns = names
    all_cleaned = cleaned.replace(-9999999.0, np.NaN).dropna(subset=["cruise"])
    time_cols = all_cleaned.columns[all_cleaned.columns.str.contains("time")]
    for col in time_cols:
        all_cleaned = all_cleaned.replace("-9999999", np.NaN).dropna(
            subset=[col]
        )  # noqa
        all_cleaned.loc[:, col] = all_cleaned[col].apply(_convert_dt)

    all_cleaned = all_cleaned.dropna(subset=time_cols).reset_index(drop="index")  # noqa
    if all_cleaned.station.isnull().values.any():
        logger.warning("NANS found in station!")
    #         all_cleaned = all_cleaned.dropna(subset=['station'])
    return all_cleaned, discrete_samples_labels


def _set_area(station):
    if isinstance(station, str):
        st = station.lower()
        if re.search(r"(oregon\s+)?slope\s+base", st):
            return "oregon-slope-base"
        elif re.search(r"axial\s+base", st):
            return "axial-base"
        elif re.search(r"axial.*international\s+district", st):
            return "axial-caldera"
        elif re.search(r"axial\s+caldera", st):
            return "axial-caldera"
        elif re.search(r"(southern\s+)?hydrate\s+ridge", st):
            return "southern-hydrate-ridge"
        elif re.search(r"mid\s+plate", st):
            return "mid-plate"
        elif re.search(r"oregon\s+inshore|ce01", st):
            return "oregon-inshore"
        elif re.search(r"oregon\s+shelf|ce02", st):
            return "oregon-shelf"
        elif re.search(r"oregon\s+offshore|ce04", st):
            return "oregon-offshore"
        elif re.search(r"washington\s+inshore|ce06", st):
            return "washington-inshore"
        elif re.search(r"washington\s+shelf|ce07", st):
            return "washington-shelf"
        elif re.search(r"washington\s+offshore|ce09", st):
            return "washington-offshore"
        else:
            raise ValueError(f"Unknown area: {st}")
    else:
        return ""


def _check_double_sensors(row: pd.Series, var: str):
    if pd.isna(row[f"{var}_1"]):
        if not pd.isna(row[f"{var}_2"]):
            return row[f"{var}_2"]
    return row[f"{var}_1"]


def parse_profile_and_discrete(
    sampledf: pd.DataFrame, array_rd: str
) -> Tuple[pd.DataFrame]:
    """
    Parses and split discrete summary data into profile and discrete.

    Parameters
    ----------
    sampledf : pd.DataFrame
        Cleaned discrete summary data
    array_rd : str
        Array reference designator

    Returns
    -------
    tuple
    """
    sampledf.loc[:, "ctd_temperature"] = sampledf.apply(
        lambda row: _check_double_sensors(row, "ctd_temperature"), axis=1
    )
    sampledf.loc[:, "ctd_conductivity"] = sampledf.apply(
        lambda row: _check_double_sensors(row, "ctd_conductivity"), axis=1
    )
    sampledf.loc[:, "ctd_salinity"] = sampledf.apply(
        lambda row: _check_double_sensors(row, "ctd_salinity"), axis=1
    )
    sampledf.loc[:, "date"] = sampledf["start_time"].apply(
        lambda row: row.strftime("%Y-%m")
    )
    sampledf.loc[:, "area_rd"] = sampledf["station"].apply(_set_area)

    profile_cols = sampledf.columns[
        sampledf.columns.str.contains("ctd|date|area_rd|cruise_id")
        & ~sampledf.columns.str.contains(
            "flag|file|bottle_closure_time|depth|latitude|longitude|beam_attenuation|oxygen_saturation|_2|_1"  # noqa
        )
    ]
    discrete_cols = sampledf.columns[
        sampledf.columns.str.contains(
            "area_rd|cruise_id|date|ctd_pressure|discrete|calculated"
        )
        & ~sampledf.columns.str.contains("flag")
    ]

    profile_df = sampledf[profile_cols]
    profile_df.loc[:, "array_rd"] = array_rd
    discrete_df = sampledf[discrete_cols]
    discrete_df.loc[:, "array_rd"] = array_rd

    return profile_df, discrete_df
