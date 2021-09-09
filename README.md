# stock-categorization

The functions in categoryandexposure.py contain various functions that relate to the categorization of stocks in a portfolio and calculate exposure to industries. To use the code, a csv file containing the portfolio information and a file containing the EOD data for all tickers in the portfolio are required.

There are multiple csv files that showcase the functions of the project.

Refer to 'sample_eod.csv' for an example of the required formatting for the EOD data.

Refer to 'sampleport.csv' for a sample portfolio that can be used as an input for the program. The portfolio file must contain the 'Symbol', 'Current Quantity', and 'Current Price' columns for all positions.

Refer to 'sectorcount.csv' for the output of the sectorCount() function. This function categories the positions in the portfolio by sector and industry.

Refer to 'index_betas.csv' for the output of the betas() function. This function calculates the market betas for each of the tickers in the portfolio against a specified series of tickers. In this example, they were compared to QQQ, SPY, IWM, DJI.

Refer to 'netbetas.csv' for the output of the netBetas() function. This function calculates the portoflio's exposure to the specified tickers.

Refer to 'residNetInv.csv' for the output of the residNetInv() function. This function removes correlation between the specified tickers and calculates the portfolio's exposure to the tickers after this adjustment. The result of this should give a more accurate idea of the actual exposure to each specified ticker.
