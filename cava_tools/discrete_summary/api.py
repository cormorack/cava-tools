from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd
from loguru import logger
from typing_extensions import Literal

from cava_tools.discrete_summary.parser import (clean_discrete_summary,
                                                get_ds_labels,
                                                get_folder_contents,
                                                parse_profile_and_discrete)
from cava_tools.discrete_summary.validator import check_types_and_replace

HERE = Path(__file__).parent
SOURCEDF = pd.read_csv(HERE.joinpath("data/source.csv"), index_col="cruise_id")
HEADERSDF = pd.read_csv(HERE.joinpath("data/discreteSummaryHeaderMap.csv"))


def get_contents(cruise_id: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Retrieves the contents from a specified cruise id

    Parameters
    ----------
    cruise_id : str
        The id of the cruise where samples come from or leave blank
        to get all the cruises.

    Returns
    -------
    pd.DataFrame
    """
    if cruise_id is None:
        df_list = []
        for cruise_id, row in SOURCEDF.iterrows():
            d = get_folder_contents(row.folder_url)
            if isinstance(d, pd.DataFrame):
                d.loc[:, "cruise_id"] = cruise_id
                df_list.append(d)
        return pd.concat(df_list)

    row = SOURCEDF.loc[cruise_id]
    contentsdf = get_folder_contents(row.folder_url)
    if isinstance(contentsdf, pd.DataFrame):
        contentsdf.loc[:, "cruise_id"] = cruise_id
        return contentsdf


def filter_contents(
    contents: pd.DataFrame, kind: Literal["readme", "summary", "all"] = "all"
) -> pd.DataFrame:
    """
    Filter content dataframe to only README and Discrete Summary CSV files.

    Parameters
    ----------
    contents : pd.DataFrame
        Contents DataFrame to be filtered down.
    kind : str
        The specific kind of files to retrieve.
        Options: 'readme', 'summary', or 'all'.
        Defaults to 'all'.

    Returns
    -------
    pd.DataFrame
    """
    filtered_files = contents[
        (contents["modified"] > "2013")
        & (contents["name"].str.contains("README|Discrete_Summary"))
        & ~(contents["name"].str.contains(".xls"))
    ].reset_index(drop="index")
    filtered_files.loc[:, "kind"] = filtered_files.apply(
        lambda row: "readme" if "README" in row["name"] else "summary", axis=1
    )
    if kind == "all":
        return filtered_files
    elif kind in ["readme", "summary"]:
        return filtered_files[filtered_files.kind.str.match(kind)].reset_index(
            drop="index"
        )
    else:
        raise ValueError(f"Unrecognized kind: {kind}")


def get_latest_content(
    contents: pd.DataFrame,
) -> Union[pd.DataFrame, pd.Series]:
    """
    Filter contents dataframe to retrieve the latest modified content

    Parameters
    ----------
    contents : pd.DataFrame
        Contents DataFrame to be filtered down.
    Returns
    -------
    pd.DataFrame or pd.Series
    """
    if "cruise_id" not in contents.columns:
        raise ValueError("Contents dataframe must have cruise_id column!")

    if "kind" in contents.columns:
        group = contents.groupby(["cruise_id", "kind"])
    else:
        group = contents.groupby("cruise_id")
    latest = contents.loc[group["modified"].idxmax()]
    if isinstance(latest, pd.DataFrame):
        latest = latest.reset_index(drop="index")
        if len(latest) == 1:
            return latest.iloc[0]
        else:
            return latest
    elif isinstance(latest, pd.Series):
        return latest


def read_and_clean(discrete_summaries: pd.DataFrame) -> Tuple[dict]:
    """
    Reads in discrete summaries csvs and cleans them by
    putting them through various validations.

    Parameters
    ----------
    discrete_summaries : pd.DataFrame
        The discrete summaries dataframe that contain url info

    Returns
    -------
    tuple
    """
    unique_kinds = discrete_summaries["kind"].unique()
    if len(unique_kinds) != 1:
        raise ValueError("Multiple kinds of files are not acceptable!")
    elif unique_kinds[0] != "summary":
        raise ValueError("Only summary files are accepted.")

    merged_summaries = pd.merge(discrete_summaries, SOURCEDF, on="cruise_id")
    expected_columns = get_ds_labels(HEADERSDF.summaryColumn)["name"]

    svdf_arrays = {}
    label_arrays = {}
    for _, row in merged_summaries.iterrows():
        logger.info("------------------------")
        logger.info(row["cruise_id"])
        url = row["url"]
        logger.info(url)
        if url.endswith(".csv"):
            svdf = pd.read_csv(url, na_values=["-9999999"])
        elif url.endswith(".xlsx"):
            svdf = pd.read_excel(url, na_values=["-9999999"])

        if row["array_rd"] not in svdf_arrays:
            svdf_arrays[row["array_rd"]] = []

        clean_svdf, discrete_sample_labels = clean_discrete_summary(
            svdf, expected_columns=expected_columns
        )
        label_arrays[row["array_rd"]] = pd.DataFrame(
            discrete_sample_labels
        ).set_index(  # noqa
            "name"
        )
        if row["array_rd"] == "CE":
            # Fix some O, 0 weirdness...
            for s in clean_svdf["station"].unique():
                if isinstance(s, str) and "O" in s:
                    logger.warning(
                        f"{s} found! Fixing to {s.replace('O', '0')}..."
                    )  # noqa
            clean_svdf.loc[:, "station"] = clean_svdf["station"].apply(
                lambda r: r.replace("O", "0") if isinstance(r, str) else r
            )
        clean_svdf.loc[:, "cruise_id"] = row["cruise_id"]
        final_svdf = clean_svdf.reset_index(drop=True)
        cleaned_final_svdf = check_types_and_replace(final_svdf)
        svdf_arrays[row["array_rd"]].append(cleaned_final_svdf)

    return {
        k: pd.concat(v, sort=False) for k, v in svdf_arrays.items()
    }, label_arrays  # noqa


def split_summary_data(
    svdf_dict: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:  # noqa
    """Split discrete summary data to profile and discrete"""
    profile_list, discrete_list = [], []
    for k, v in svdf_dict.items():
        sampledf = v.copy()
        profile_df, discrete_df = parse_profile_and_discrete(sampledf, k)
        profile_list.append(profile_df)
        if any(discrete_df.columns.isin(["calculated_dic", "calculated_pco2"])):  # noqa
            if all(discrete_df["calculated_dic"].isna()):
                discrete_df.drop("calculated_dic", axis=1, inplace=True)
            if all(discrete_df["calculated_pco2"].isna()):
                discrete_df.drop("calculated_pco2", axis=1, inplace=True)
        discrete_list.append(discrete_df)

    all_profiles = pd.concat(profile_list, sort=False).reset_index(drop=True)
    all_discrete = pd.concat(discrete_list, sort=False).reset_index(drop=True)
    return {"profile": all_profiles, "discrete": all_discrete}
