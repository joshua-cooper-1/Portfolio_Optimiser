import yfinance as yf
import math
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt






def calculate_risk(weights, cov_matrix):
    """
    Calculates the portfolio risk (volatility) given asset weights and the covariance matrix.

    This computes the portfolio standard deviation using the quadratic form:
        sqrt(wᵀ Σ w)

    Parameters
    ----------
    weights : array-like
        Portfolio weights of the assets. Must sum to 1 for a fully invested portfolio.
    cov_matrix : pandas.DataFrame
        Covariance matrix of asset returns (annualised).

    Returns
    -------
    float
        The portfolio risk (annualised standard deviation).
    """
    weights = np.array(weights)
    return np.sqrt(weights.T @ cov_matrix.values @ weights)

def calculate_return(weights, expected_returns):
    """
    Calculates the expected portfolio return.

    Parameters
    ----------
    weights : array-like
        Portfolio weights for each asset (must sum to 1).
    expected_returns : array-like
        Expected return of each asset.

    Returns
    -------
    float
        Expected portfolio return.
    """
    return np.dot(weights, expected_returns)



'''
Not using fully expanded version as matrix version is more computationally efficient
def calculate_risk(weights, cov_matrix, tickers):
    """
    Calculates the portfolio risk given asset weights, covariance matrix,
    and corresponding tickers.

    This explicitly computes the portfolio variance as the double summation:
        Σ Σ (wᵢ * wⱼ * σᵢⱼ)
    where:
        - wᵢ, wⱼ are asset weights
        - σᵢⱼ is the covariance between assets i and j

    Parameters
    ----------
    weights : array-like
        Portfolio weights of the assets. must sum to 1 for a fully invested portfolio.
    cov_matrix : pandas.DataFrame
        Covariance matrix of asset returns (annualised), indexed by tickers.
    tickers : list of str
        List of asset tickers corresponding to the weights.

    Returns
    -------
    float
        The portfolio risk (annualised standard deviation).
    """
    total_risk = 0
    for i in range(len(weights)):
        for j in range(len(weights)):
            w_i = weights[i]
            w_j = weights[j]
            ticker_i = tickers[i]
            ticker_j = tickers[j]
            
            total_risk += w_i * w_j * cov_matrix.loc[ticker_i, ticker_j]

    return math.sqrt(total_risk)
'''

'''
Again not using as numpy is more efficient
def calculate_return(weights, expected_returns):
    """
    Calculate the expected portfolio return given asset weights and expected returns.

    This explicitly computes the weighted sum:
        R_p = Σ (wᵢ * μᵢ)
    where:
        - wᵢ is the portfolio weight of asset i
        - μᵢ is the expected return of asset i

    Parameters
    ----------
    weights : array-like
        Portfolio weights of the assets, which must sum to 1 for a fully invested portfolio.
    expected_returns : array-like
        Expected returns of the assets (annualised or daily).

    Returns
    -------
    float
        The expected portfolio return.
    """
    total = 0
    for i in range(len(weights)):
        weight = weights[i]
        expected_return = expected_returns[i]
        total += weight * expected_return

    return total
'''


def optimise_portfolio(data, tickers, range_of_returns):
    """
    Computes the efficient frontier for a set of assets using mean variance optimisation.

    This function estimates expected returns and the covariance matrix from historical 
    price data, then uses constrained optimisation (SLSQP) to find portfolio weights 
    that minimise risk for a range of target returns.

    Parameters
    ----------
    data : pandas.DataFrame
        Historical adjusted close prices with tickers as columns
    tickers : list of str
        List of asset tickers to include in the portfolio.
    range_of_returns : list or tuple of float
        Lower and upper bounds of expected returns (annualised) for which to compute 
        the efficient frontier, like [0.05, 0.25] for example.

    Returns
    -------
    list of tuples
        Each tuple has the form (risk, return, weights), where:
        - risk : float
            Portfolio volatility (annualised standard deviation).
        - return : float
            Target expected return (annualised).
        - weights : numpy.ndarray
            Optimal asset weights achieving the target return.

    Notes
    -----
    - Expected returns are annualised using compound returns:
        (1 + daily_mean_return)^252 - 1
    - Portfolio risk is computed as the square root of the quadratic form:
        σ_p = sqrt(wᵀ Σ w)
    - Optimisation constraints:
        * Weights sum to 1
        * Expected return equals the current target return
        * Weights are bounded between 0 and 1 (assumed no short selling)
    - Historical prices are assumed to be daily; 252 trading days are used for annualisation.
    """

    daily_returns = data.pct_change().dropna()
    cov_matrix = daily_returns.cov() * 252

    mean = daily_returns.mean()
    expected_returns = (1 + mean)**252 - 1  # expected_returns annualised

    bounds = [(0, 1) for _ in range(len(tickers))]

    efficient_portfolios = []

    min_return = range_of_returns[0]
    max_return = range_of_returns[1]
    target_returns = np.linspace(min_return, max_return, 100)

    x0 = np.ones(len(tickers)) / len(tickers)  # equal weight start

    for target_return in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # weights sum to 1
            {'type': 'eq', 'fun': lambda w, tr=target_return: calculate_return(w, expected_returns) - tr}
        )

        result = minimize(fun=lambda w: calculate_risk(w, cov_matrix),
                          x0=x0, method='SLSQP', bounds=bounds, constraints=constraints)

        weights = result.x
        optimal_risk = calculate_risk(weights, cov_matrix)
     
        efficient_portfolios.append((optimal_risk, target_return, weights))

    return efficient_portfolios


def plot_efficient_frontier(efficient_portfolios):
    """
    Plot the efficient frontier from a list of optimised portfolios.

    Parameters
    ----------
    efficient_portfolios : list of tuples
        Each tuple should be of the form (risk, return, weights), 
        where:
        - risk : float
            Portfolio volatility (standard deviation, annualised).
        - return : float
            Portfolio expected return (annualised).
        - weights : numpy.ndarray
            Optimal asset weights corresponding to the risk-return pair.

    Notes
    -----
    - The x-axis shows portfolio risk (volatility).
    - The y-axis shows expected portfolio return.
    - The curve typically exhibits the "Markowitz bullet" shape, 
      representing the trade-off between risk and return.
    """
    plt.figure(figsize=(12, 8))  # width=12 inches, height=8 inches
    x, y, _ = zip(*efficient_portfolios)  # unzip the tuples, weights not needed
    plt.plot(x, y, color='green', linestyle='-', label='Efficient Frontier')
    plt.xlabel('Risk')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.show()

def plot_efficient_frontier_with_individual_stocks(data, efficient_portfolios, tickers):
    """
    Plot the efficient frontier alongside individual stock risk-return points.

    This function produces a graph where:
    - The efficient frontier is shown as a green line.
    - Individual stocks are plotted as scatter points with labels.
    
    Risk is represented on the x-axis (annualised standard deviation),
    and expected return is represented on the y-axis (annualised).

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame of historical stock prices with tickers as columns.
    efficient_portfolios : list of tuples
        Each tuple should contain (risk, expected_return, weights) for a portfolio.
    tickers : list of str
        List of stock tickers to include as individual assets.

    Returns
    -------
    None
        Displays the plot; does not return a value.
    """

    daily_returns = data.pct_change().dropna()


    plt.figure(figsize=(12, 8))  # width=12 inches, height=8 inches
    for ticker in tickers:
        risk = np.sqrt(daily_returns[ticker].var() * 252)


        mean = daily_returns[ticker].mean()
        expected_return = (1+mean)**252 -1
        plt.scatter(risk, expected_return, s=100, alpha=0.7, label=ticker)
    
 
        plt.annotate(ticker, (risk, expected_return), xytext=(5, 5), 
                textcoords='offset points', fontsize=9)


    x, y, _ = zip(*efficient_portfolios)  # unzip the tuples, weights not needed
    plt.plot(x, y, color='green', linestyle='-', label='Efficient Frontier')
    plt.xlabel('Risk')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.show()

    



tickers = ['AAPL', 'MSFT', 'GOOG'] #change which stocks to see

data = yf.download(tickers, period="5y", auto_adjust=False)["Adj Close"] #adjusted close prices for a period of 5 years
data = data[tickers] 
efficient_portfolios = optimise_portfolio(data, tickers, [0.2, 0.3]) #adjust



#plot_efficient_frontier(efficient_portfolios)
plot_efficient_frontier_with_individual_stocks(data, efficient_portfolios, tickers)








