import numpy as np
from scipy.stats import t
import math
import statsmodels.api as sm
from matplotlib import pyplot as plt


def conf_interval(data: np.ndarray, conf_level: float = 0.95, loc: bool = False, scale: bool = False) -> (float, float):
    # ASSUMPTION: normal distribution, which can be verified using Q-Q Plot
    #fig = sm.qqplot(data, line="q")
    #plt.show()

    df = data.size - 1
    alpha = (1 - conf_level)/2

    mean = np.mean(data)
    std = np.std(data)

    if (not loc) and (not scale):
        t_param = t.ppf(1-alpha, df=df)
    elif loc and scale:
        t_param = t.ppf(1 - alpha, df=df, loc=mean, scale=std)
    elif loc:
        t_param = t.ppf(1 - alpha, df=df, loc=mean)
    elif scale:
        t_param = t.ppf(1 - alpha, df=df, scale=std)

    conf = (t_param * std / math.sqrt(data.size))
    lower_end = mean - conf
    upper_end = mean + conf
    return (lower_end, upper_end)