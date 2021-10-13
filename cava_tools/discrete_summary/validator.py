import numpy as np
import pandas as pd
from loguru import logger


def check_name(name: str) -> str:
    """
    Checks discrete sample column name for any errors
    """
    low_name = name.lower()
    if "fluorescense" in low_name or "flourescence" in low_name:
        logger.warning(
            f"Fluorescence is misspelled in {name}... replacing to Fluorescence"  # noqa
        )
        new_name = name.replace("fluorescense".title(), "fluorescence".title())
        new_name = new_name.replace(
            "flourescence".title(), "fluorescence".title()
        )  # noqa
    elif "start positioning" in low_name:
        logger.warning(
            f"start positioning found in {name}... replacing to Start Position"
        )
        new_name = "Bottom Depth at Start Position"
    elif "phanalysis" in low_name:
        logger.warning(f"pH Analysis is strung together in {name}... fixing...")  # noqa
        new_name = name.replace("pHAnalysis", "pH Analysis")
    else:
        new_name = name
    return new_name


def check_types_and_replace(df: pd.DataFrame) -> pd.DataFrame:
    """Checking for invalid data types and replace them"""
    value_str = ["ctd", "discrete", "calculated"]
    for k, v in df.dtypes.items():
        if (
            any(x in k for x in value_str)
            and "file" not in k
            and "bottle_closure_time" not in k
            and "flag" not in k
        ):
            if v == "O":
                do = df[k]
                string_filter = do.str.contains("\w").fillna(False)  # noqa
                invalid_values = ",".join(do[string_filter].unique())
                logger.warning(
                    f"** {k} ** contains invalid float values: {invalid_values} \n\tReplacing invalid values with NaNs..."  # noqa
                )
                # Replace invalid values with NaNs
                df.loc[string_filter, k] = np.NaN
                # Set final dtype to float64
                df.loc[:, k] = df[k].astype(np.float64)
    return df
