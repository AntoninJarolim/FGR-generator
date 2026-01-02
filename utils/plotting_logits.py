import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_series(series_with_names):
    """
    Plot any number of equal-length series against x = range(len)
    using seaborn with different hues.

    Args:
        series_with_names: list of (series, name) pairs
    """
    if not series_with_names:
        raise ValueError("Input list is empty")

    lengths = [len(series) for series, _ in series_with_names]
    if len(set(lengths)) != 1:
        raise ValueError(f"All series must have the same length, got {lengths}")

    n = lengths[0]

    df = pd.DataFrame({
        "x": list(range(n)) * len(series_with_names),
        "value": [
            val
            for series, _ in series_with_names
            for val in series
        ],
        "series": [
            name
            for _, name in series_with_names
            for _ in range(n)
        ],
    })

    sns.lineplot(
        data=df,
        x="x",
        y="value",
        hue="series",
        style="series",
        markers=True,
        dashes=False
    )
    plt.show()



