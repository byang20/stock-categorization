
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from nelson_siegel_svensson import NelsonSiegelSvenssonCurve, NelsonSiegelCurve
from nelson_siegel_svensson.calibrate import calibrate_nss_ols,calibrate_ns_ols
from scipy.optimize import lsq_linear

#set of indexes that all calculations will be made against
#indexTickers = ['QQQ','SPY','DIA','IWM','XLF','IAT','IAI','VGT']
indexTickers = ['QQQ','SPY','DIA','IWM']
#EOD data of all tickers
eodDataFile = 'EOD_20210527.csv'

#divides all stocks in portfolio by sector and industry
#filename: filename of portfolio
#resultfile: filename to save results to
def sectorCount(filename,resultfile):
    results = pd.DataFrame(columns = ['sector','industry','total','tickers'])
    portfolio = pd.read_csv(filename)

    categoryData = pd.read_csv('quandldata.csv')
    overwriteData = pd.read_csv('overwrites.csv')

    for index, row in portfolio.iterrows():
        symbol = row['Symbol']
        currPrice = row['Current Price']
        
        totalInv = 0
        if isinstance(currPrice,str):
            if(currPrice[-1]=='M'):
                currPrice = float(currPrice[:currPrice.index('.')+3].replace(',',''))
                totalInv = currPrice*row['Current Quantity']
            else:
                totalInv = float(currPrice.replace(',',''))*float(row['Current Quantity'])
        else:
            if(isinstance(row['Current Price'],float)):
                totalInv = float(row['Current Quantity']) * float(row['Current Price'])
            else:
                totalInv = float(row['Current Quantity']) * float(row['Current Price'].replace(',','')) #quant is int, price is string???

        currStockRow = overwriteData.loc[overwriteData['ticker']==symbol]
        if currStockRow.empty: 
            currStockRow = categoryData.loc[categoryData['ticker']==symbol]
        if currStockRow.empty:
            print(symbol , " could not be found in categories")
            continue

        currency = currStockRow['currency'].values[0]
        currSector = currStockRow['sicsector'].values[0]
        currIndustry = currStockRow['sicindustry'].values[0]

        if currency != 'USD' and not pd.isna(currency):
            currSector = 'International Securities'
        
        resultRow = results.loc[(results['sector']==currSector) & (results['industry']==currIndustry)]
        if resultRow.empty:
            newLine = {'sector': currSector,'industry': currIndustry, 'total': totalInv, 'tickers': symbol}
            results = results.append(newLine, ignore_index=True)
        else:
            results.loc[(results['sector']==currSector) & (results['industry']==currIndustry),'total'] += totalInv
            results.loc[(results['sector']==currSector) & (results['industry']==currIndustry),'tickers'] += "," + symbol

    pd.options.display.float_format = '${0:,.0f}'.format

    shorts = results.loc[(results['total'])<0].sort_values(['sector','industry'])
    longs = results.loc[(results['total'])>=0].sort_values(['sector','industry'])
    pd.concat([shorts,longs]).to_csv(resultfile)
    print('Short Total: ')
    print(shorts.iloc[:,2:3].sum())
    print('Long Total: ')
    print(longs.iloc[:,2:3].sum())
    print('Portfolio Net: ')
    print(results.iloc[:,2:3].sum())

end = '2021-5-27'
start1m = '2021-4-27'
start3m = '2021-2-27'
start1y = '2020-5-27'

#calculates betas for each ticker in portfolio against specified indexes for 3 month and 1 year
#filename: file name of portfolio
#resultfile: file name to save results to
#portfolioEODFile: EOD file containing data for tickers in portfolio, defaults to full EOD file
def betas(filename,resultfile, portfolioEODFile = eodDataFile):
    results = pd.DataFrame(columns = ['ticker'])
    for index in indexTickers:
        results[index+'3m_b'] = None
        results[index+'3m_rsq'] = None
    for index in indexTickers:
        results[index+'1y_b'] = None
        results[index+'1y_rsq'] = None

    portfolio = pd.read_csv(filename)
    peod = pd.read_csv(portfolioEODFile)
    peod['date'] = pd.to_datetime(peod['date'])
    indexes = pd.read_csv('indexes.csv')
    indexes['date'] = pd.to_datetime(indexes['date'])

    for index in indexTickers:
        indexes.loc[indexes['ticker']==index,'close'] = indexes.loc[indexes['ticker']==index,'close'].pct_change()
    indexData3m = pd.DataFrame()
    indexData1y = pd.DataFrame()

    for indexTicker in indexTickers:
        curr3m = indexes.loc[((indexes['date']>start3m)&(indexes['date']<=end)) & (indexes['ticker']==indexTicker)]
        indexData3m[indexTicker+'dates'] = curr3m['date'].values
        indexData3m[indexTicker] = curr3m['close'].values.reshape(-1,1)
        curr1y = indexes.loc[((indexes['date']>start1y)&(indexes['date']<=end)) & (indexes['ticker']==indexTicker)]
        indexData1y[indexTicker+'dates'] = curr1y['date'].values
        indexData1y[indexTicker] = curr1y['close'].values.reshape(-1,1)
    
    for index, row in portfolio.iterrows():
        symbol = row['Symbol']
        currStock = peod.loc[(peod['ticker']==symbol)]
        currStock = currStock.copy()
        currStock['close'] = currStock['close'].pct_change()

        symbolPrices3m = currStock.loc[((currStock['date']>start3m)&(currStock['date']<=end))]
        symbolPrices3m=symbolPrices3m.drop_duplicates()
        symbolPrices1y = currStock.loc[((currStock['date']>start1y)&(currStock['date']<=end))]
        symbolPrices1y=symbolPrices1y.drop_duplicates()

        if(symbolPrices3m.empty):
            print(symbol + " could not be found in portfolioeod")
            continue
        
        if (len(symbolPrices3m) < len(indexData3m)) or (len(symbolPrices1y) < len(indexData1y)):
            print('Not enough data for: ' + symbol)
            continue

        mismatch = 0
        count = 0

        for index1, row1 in symbolPrices3m.iterrows():
            for indexTicker in indexTickers:
                if row1['date'] != indexData3m.iloc[count][indexTicker+'dates']:
                    print('3m Date mismatch for ', symbol)
                    mismatch=1
                    break
            if mismatch:
                break
            count = count+1

        count = 0
        for index1, row1 in symbolPrices1y.iterrows():
            for indexTicker in indexTickers:
                if row1['date'] != indexData1y.iloc[count][indexTicker+'dates']:
                    print('1y Date mismatch for ', symbol)
                    mismatch=1
                    break
            if mismatch:
                break
            count = count+1

        if mismatch:
            continue

        x3m = symbolPrices3m['close'].values.reshape(-1,1)
        x1y = symbolPrices1y['close'].values.reshape(-1,1)

        linRegModels = pd.DataFrame()
        for indexTicker in indexTickers:
            linRegModels[indexTicker+'3mmodel'] = np.array([LinearRegression().fit(indexData3m[indexTicker].values.reshape(-1,1),x3m)])
            linRegModels[indexTicker+'1ymodel'] = LinearRegression().fit(indexData1y[indexTicker].values.reshape(-1,1),x1y)

        s1 = pd.Series([symbol], index=['ticker'])
        for indexTicker in indexTickers:
            s2 = pd.Series([linRegModels[indexTicker+'3mmodel'][0].coef_.item()],index=[indexTicker+'3m_b'])
            s3 = pd.Series([linRegModels[indexTicker+'3mmodel'][0].score(indexData3m[indexTicker].values.reshape(-1,1),x3m).item()],index=[indexTicker+'3m_rsq'])
            s4 = pd.Series([linRegModels[indexTicker+'1ymodel'][0].coef_.item()],index=[indexTicker+'1y_b'])
            s5 = pd.Series([linRegModels[indexTicker+'1ymodel'][0].score(indexData1y[indexTicker].values.reshape(-1,1),x1y).item()],index=[indexTicker+'1y_rsq'])
            s1 = s1.append([s2,s3,s4,s5])

        results = results.append(s1, ignore_index = True)

        results.to_csv(resultfile)

#calculate beta between the indexes for 3 month and 1 year
def indexBetas(indexEODFile,resultfile):
    results = pd.DataFrame()
    for ticker in indexTickers:
        results = results.append([ticker])

    results.columns=['Symbol']
    results.to_csv('indextickers.csv')

    betas('indextickers.csv',resultfile,indexEODFile)

#calculate portfolio exposure to each industry and sector 
#portfoliofile: file name of portfolio
#portcategories: resultfile of sectorCount()
#betas: resultfile of betas()
#resultfile: file name to save results to
def netBetas(portfoliofile,portcategories,betas,resultfile):
    results = pd.DataFrame(columns = ['ticker','sector','industry'])
    for index in indexTickers:
        results[index+'3m_b'] = None
        results[index+'3m_rsq'] = None
    for index in indexTickers:
        results[index+'1y_b'] = None
        results[index+'1y_rsq'] = None
    results['totalInv'] = None
    results['absInv'] = None

    portfolio = pd.read_csv(portfoliofile)
    categories = pd.read_csv(portcategories)
    betas = pd.read_csv(betas)

    for index, row in betas.iterrows():
        symbol = row['ticker']
        portfolioRow = portfolio.loc[portfolio['Symbol'] == symbol]
        categoryRow = categories.loc[categories['tickers'].str.contains(symbol)]

        if categoryRow.empty:
            print(symbol, 'not found in categories file ', portcategories)
            continue

        if(isinstance(portfolioRow['Current Price'],float)):
            totalInv = float(portfolioRow['Current Quantity']) * float(portfolioRow['Current Price'])
        else:
            try:
                price = portfolioRow['Current Price'].iloc[0]
            except:
                print(portfolioRow)
                print((portfolioRow['Current Price']))
                exit()
            totalInv = float(portfolioRow['Current Quantity']) * float(price.replace(',','')) #quant is int, price is string???

        currSector = categoryRow['sector'].values[0]
        currIndustry = categoryRow['industry'].values[0]

        categorymask = (results['sector']==currSector) & (results['industry']==currIndustry)
        resultRow = results.loc[categorymask]
        if resultRow.empty:
            s1 = pd.Series([symbol,currSector,currIndustry],index=['ticker','sector','industry'])
            for indexTicker in indexTickers:
                s1 = s1.append(pd.Series(row[indexTicker+'3m_b']*totalInv,index = [indexTicker+'3m_b']))
                s1 = s1.append(pd.Series(row[indexTicker+'3m_rsq'],index = [indexTicker+'3m_rsq']))
            for indexTicker in indexTickers:
                s1 = s1.append(pd.Series(row[indexTicker+'1y_b']*totalInv,index = [indexTicker+'1y_b']))
                s1 = s1.append(pd.Series(row[indexTicker+'1y_rsq'],index = [indexTicker+'1y_rsq']))
            s1 = s1.append( pd.Series([totalInv,abs(totalInv)],index=['totalInv','absInv']))
            results = results.append(s1, ignore_index = True)
        else:
            results.loc[categorymask,'totalInv'] += totalInv
            totalInv = abs(totalInv)
            prevAbsInv = resultRow['absInv'].values[0]
            newAbsInv = prevAbsInv+totalInv          

            results.loc[categorymask,'ticker'] += "," + symbol
            for indexTicker in indexTickers:
                results.loc[categorymask,indexTicker+'3m_b'] += row[indexTicker+'3m_b']*totalInv
                results.loc[categorymask,indexTicker+'3m_rsq'] = (row[indexTicker+'3m_rsq']*totalInv+results.loc[categorymask,indexTicker+'3m_rsq']*prevAbsInv)/newAbsInv
                results.loc[categorymask,indexTicker+'1y_b'] += row[indexTicker+'1y_b']*totalInv
                results.loc[categorymask,indexTicker+'1y_rsq'] = (row[indexTicker+'1y_rsq']*totalInv+results.loc[categorymask,indexTicker+'1y_rsq']*prevAbsInv)/newAbsInv
                results.loc[categorymask,'absInv'] += totalInv
    shorts = results.loc[(results['totalInv'])<0].sort_values(['sector','industry'])
    longs = results.loc[(results['totalInv'])>=0].sort_values(['sector','industry'])
    pd.concat([shorts,longs]).to_csv(resultfile)
    results.to_csv(resultfile)

#removes correlation from each ticker with all indexes but one, correlates residual with the last ticker
#portfoliofile: file name of portfolio
#indexBetasFile: resultfile of indexBetas()
#resultfile: file name to save results to
#portfolioEODFile: EOD file containing data for tickers in portfolio, defaults to full EOD file
#indexesfile: EOD file containing data for indexes, defaults to full EOD file
def cov(portfoliofile,indexBetasFile,resultfile,portfolioEODFile=eodDataFile,indexesfile=eodDataFile):
    mainindex = indexTickers[0]
    portfolio = pd.read_csv(portfoliofile)
    indexes = pd.read_csv(indexesfile)
    indexBetas = pd.read_csv(indexBetasFile)

    indexes['date'] = pd.to_datetime(indexes['date'])
    for indexTicker in indexTickers:
        indexes.loc[indexes['ticker']==indexTicker,'close'] = indexes.loc[indexes['ticker']==indexTicker,'close'].pct_change()
    
    mask3m = (indexes['date']>start3m) & (indexes['date']<=end)
    mask1y = (indexes['date']>start1y) & (indexes['date']<=end)

    indexBetas3m = pd.Series(dtype='float64')
    for indexTicker in indexTickers:
        if indexTicker != mainindex:
            indexBetas3m = indexBetas3m.append(pd.Series([float(indexBetas.loc[indexBetas['ticker']==indexTicker,mainindex+'3m_b'].values[0])],index=[indexTicker]))

    
    indexResids3m = pd.DataFrame()
    for indexTicker in indexTickers:
        if indexTicker != mainindex:
            indexResids3m[indexTicker] = indexes.loc[mask3m & (indexes['ticker']==indexTicker),'close'].to_numpy()-indexBetas3m[indexTicker]*(indexes.loc[mask3m & (indexes['ticker']==mainindex),'close'].to_numpy())
    
    mainIndex3m = indexes.loc[mask3m & (indexes['ticker']==mainindex),'close'].to_numpy()
    data3mresid = np.array([mainIndex3m])
    data3mresid = np.append(data3mresid,np.transpose(indexResids3m.to_numpy()),axis=0)

    indexBetas1y = pd.Series(dtype='float64')
    for indexTicker in indexTickers:
        if indexTicker != mainindex:
            indexBetas1y = indexBetas1y.append(pd.Series([float(indexBetas.loc[indexBetas['ticker']==indexTicker,mainindex+'1y_b'].values[0])],index=[indexTicker]))
    
    indexResids1y = pd.DataFrame()
    for indexTicker in indexTickers:
        if indexTicker != mainindex:
            indexResids1y[indexTicker] = indexes.loc[mask1y & (indexes['ticker']==indexTicker),'close'].to_numpy()-indexBetas1y[indexTicker]*(indexes.loc[mask1y & (indexes['ticker']==mainindex),'close'].to_numpy())
    
    mainIndex1y = indexes.loc[mask1y & (indexes['ticker']==mainindex),'close'].to_numpy()

    data1yresid = np.array([mainIndex1y])
    data1yresid = np.append(data1yresid,np.transpose(indexResids1y.to_numpy()),axis=0)

    peod = pd.read_csv(portfolioEODFile)
    peod['date'] = pd.to_datetime(peod['date'])

    results = pd.DataFrame(columns = ['ticker'])
    results[mainindex+'3m_b'] = None
    results[mainindex+'3m_netb'] = None
    for index in indexTickers:
        results[index+'3m_b'] = None
    results['3m_rsq'] = None
    results[mainindex+'1y_b'] = None
    results[mainindex+'1y_netb'] = None
    for index in indexTickers:
        results[index+'1y_b'] = None
    results['1y_rsq'] = None
    
    for index, row in portfolio.iterrows():
        symbol = row['Symbol']
        currStock = peod.loc[(peod['ticker']==symbol)]
        currStock = currStock.copy()
        currStock['close'] = currStock['close'].pct_change()

        symbolPrices3m = currStock.loc[((currStock['date']>start3m)&(currStock['date']<=end))]
        symbolPrices3m = symbolPrices3m.drop_duplicates()
        symbolPrices1y = currStock.loc[((currStock['date']>start1y)&(currStock['date']<=end))]
        symbolPrices1y = symbolPrices1y.drop_duplicates()

        y3m = symbolPrices3m['close'].values.reshape(-1,1)
        y1y = symbolPrices1y['close'].values.reshape(-1,1)
        
        if len(symbolPrices3m)<len(mainIndex3m) or len(symbolPrices1y)<len(mainIndex1y):
            print('not enough data on: ', symbol)
            continue

        lb = np.array([-np.inf,-0.5])
        ub = np.array([np.inf,0.5])
        for i in range (2,len(indexTickers)):
            lb = np.append(lb,-0.7)
            ub = np.append(ub,0.7)

        res3m = lsq_linear(np.transpose(data3mresid),y3m.flatten(),bounds=(lb,ub))
        res1y = lsq_linear(np.transpose(data1yresid),y1y.flatten(),bounds=(lb,ub))

        s1 = pd.Series([symbol],index=['ticker'])

        s1 = s1.append(pd.Series(res3m.x[0],index=[mainindex+'3m_b']))
        netb3m = res3m.x[0]
        for i in range(1,len(indexTickers)):
            netb3m -= float(indexBetas.loc[indexBetas['ticker']==indexTickers[i],mainindex+'3m_b'].values[0])*res3m.x[i]
        s1 = s1.append(pd.Series([netb3m],index=[mainindex+'3m_netb']))
        for i in range(1,len(indexTickers)):
            s1 = s1.append(pd.Series([res3m.x[i]],index=[indexTickers[i]+'3m_b']))
        s1 = s1.append(pd.Series([1-(np.sum(np.square(res3m.fun)))/(np.sum(np.square(y3m.flatten()-np.mean(y3m))))],index=['3m_rsq']))

        s1 = s1.append(pd.Series(res1y.x[0],index=[mainindex+'1y_b']))
        netb1y = res1y.x[0]
        for i in range(1,len(indexTickers)):
            netb1y -= float(indexBetas.loc[indexBetas['ticker']==indexTickers[i],mainindex+'1y_b'].values[0])*res1y.x[i]
        s1 = s1.append(pd.Series([netb1y],index=[mainindex+'1y_netb']))
        for i in range(1,len(indexTickers)):
            s1 = s1.append(pd.Series([res1y.x[i]],index=[indexTickers[i]+'1y_b']))
        s1 = s1.append(pd.Series([1-(np.sum(np.square(res1y.fun)))/(np.sum(np.square(y1y.flatten()-np.mean(y1y))))],index=['1y_rsq']))

        results = results.append(s1,ignore_index=True)


    results.to_csv(resultfile)

#calculates the exposure of the portfolio to each index after removing correlation from all other indexes, categorized by industry and sector
#residfile: resultfile of cov()
#portfoliofile: file name of portfolio
#portcategories: result file of sectorCount()
#resultfile: file name to save results to
def residNetInv(residfile,portfoliofile,portcategories,resultfile):
    portfolio = pd.read_csv(portfoliofile)
    portfolio = portfolio.drop_duplicates()
    categories = pd.read_csv(portcategories)
    resid = pd.read_csv(residfile)
    columns = list(resid)
    columns = columns[2:]
    
    results = pd.DataFrame(columns=['ticker','sector','industry'])
    for column in columns:
        results[column] = None
    results['totalInv'] = None
    results['absInv'] = None

    for index,row in resid.iterrows():
        symbol = row['ticker']
        portfolioRow = portfolio.loc[portfolio['Symbol'] == symbol]
        categoryRow = categories.loc[categories['tickers'].str.contains(symbol)]

        if categoryRow.empty:
            print(symbol, 'not found in categories file ', portcategories)
            continue

        currPrice = portfolioRow['Current Price'].item()

        if isinstance(currPrice,str):
            if(currPrice[-1]=='M'):
                currPrice = float(currPrice[:currPrice.index('.')+3].replace(',',''))
                totalInv = currPrice*portfolioRow['Current Quantity']
            else:
                totalInv = float(currPrice.replace(',',''))*portfolioRow['Current Quantity']
        else:
            if(isinstance(portfolioRow['Current Price'],float)):
                totalInv = float(portfolioRow['Current Quantity']) * float(portfolioRow['Current Price'])
            else:
                price = portfolioRow['Current Price'].item()
                totalInv = float(portfolioRow['Current Quantity']) * float(price.replace(',','')) #quant is int, price is string???
        totalInv = totalInv.item()
        
        currSector = categoryRow['sector'].values[0]
        currIndustry = categoryRow['industry'].values[0]

        categorymask = (results['sector']==currSector) & (results['industry']==currIndustry)
        resultRow = results.loc[categorymask]

        if resultRow.empty:
            s1 = pd.Series([symbol,currSector,currIndustry],index=['ticker','sector','industry'])
            for column in columns:
                if column[-3:] == 'rsq':
                    s1 = s1.append(pd.Series(row[column],index = [column]))
                else:
                    s1 = s1.append(pd.Series(row[column]*totalInv,index = [column]))
            s1 = s1.append( pd.Series([totalInv,abs(totalInv)],index=['totalInv','absInv']))
            results = results.append(s1, ignore_index = True)
        else:
            results.loc[categorymask,'totalInv'] += totalInv
            totalInv = abs(totalInv)
            prevAbsInv = resultRow['absInv'].values[0]
            newAbsInv = prevAbsInv+totalInv
            results.loc[categorymask,'ticker'] += "," + symbol
            for column in columns:
                if column[-3:] == 'rsq':
                    results.loc[categorymask,column] = (row[column]*totalInv+results.loc[categorymask,column]*prevAbsInv)/newAbsInv
                else:
                    results.loc[categorymask,column] += row[column]*totalInv
            results.loc[categorymask,'absInv'] += totalInv
    
    shorts = results.loc[(results['totalInv'])<0].sort_values(['sector','industry'])
    longs = results.loc[(results['totalInv'])>=0].sort_values(['sector','industry'])
    pd.concat([shorts,longs]).to_csv(resultfile)

#finds stocks in portfolio that don't have categories(need to be overwritten)
def getOverwrite(filename):
    portfolio = pd.read_csv(filename)

    categoryfile = 'quandldata.csv'
    categoryData = pd.read_csv(categoryfile)

    toOverwrite = pd.DataFrame()

    for index, row in portfolio.iterrows():
        symbol = row['Symbol']        
        try:
            currStockRow = categoryData.loc[categoryData['ticker']==symbol]
            currSector = currStockRow['sicsector'].values[0]
            currIndustry = currStockRow['sicindustry'].values[0]
        except:
            print(symbol , " could not be found in quandl category data")
            continue

        if pd.isna(currSector) or pd.isna(currIndustry):
            toOverwrite = toOverwrite.append(currStockRow, ignore_index=True)
    
    toOverwrite.to_csv('to_overwrite.csv')

#get data for the indexes
def get_indexes():
    
    eod = pd.read_csv(eodDataFile)
    result = pd.DataFrame()
    
    for ticker in indexTickers:
        curr = eod.loc[eod.iloc[:, 0]==ticker]
        result = result.append(curr)

    result.columns = ['ticker','date','open','high','low','close','volume','dividend','split','adj_open','adj_high','adj_low','adj_close','adj_volume']
    result.to_csv('indexes.csv')

#gathers eod data for stocks in portfolio
def get_portfolioeod(portfoliofile,outputfile):
    #peod = pd.read_csv(outputfile)
    eod = pd.read_csv(eodDataFile)
    results = pd.DataFrame()
    portfolio = pd.read_csv(portfoliofile)

    for index, row in portfolio.iterrows():
        symbol = row['Symbol']
        curr = eod.loc[eod.iloc[:, 0]==symbol]
        results = pd.concat([results,curr])

    results.columns = ['ticker','date','open','high','low','close','volume','dividend','split','adj_open','adj_high','adj_low','adj_close','adj_volume']
    results.to_csv(outputfile)

#get certain piece of data(for debugging)
def get_data():
    file = pd.read_csv('indexes.csv')
    ticker = 'XLF'
    start = '1990-1-8'
    end = '2021-6-11'

    data = file.loc[(file['ticker']==ticker)&(file['date']>=start)&(file['date']<=end),'close']
    data.to_csv('tickerdata.csv')



def main():
    portfolio = 'sampleport.csv'

    #get_data()
    #svd('dailytreasury.csv')

    #get_indexes()
    #yieldvsindex('yieldsvdnodiff.csv','indexes.csv')

    #getOverwrite(portfolio)

    #get_portfolioeod(portfolio,'MSportfolioeod.csv')
    #indexBetas('indexes.csv','indexbetas.csv')

    #sectorCount(portfolio,'MSsectorcount.csv')
    #betas(portfolio,'MSbetas.csv','MSportfolioeod.csv')
    #netbetas(portfolio,'sectorcount.csv', 'MSbetas.csv','netbetas.csv')
    #cov(portfoliofile,indexBetasFile,resultfile,portfolioEODFile=eodDataFile,indexesfile=eodDataFile)
    #cov(portfolio,'indexbetas.csv','cov.csv', 'MSportfolioeod.csv','indexes.csv')
    residNetInv('cov.csv',portfolio,'MSsectorcount.csv','residNetInv.csv')

if __name__ == '__main__':
    main()


