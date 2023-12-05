"""
Created on Sat Nov 25 03:59:37 2023

@author: jxzou
"""

import pandas as pd
import datetime as dt
import numpy as np
import scipy.stats as st
from math import exp
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
import math



def rounding(l):
    
    return [round(ele, 4) for ele in l]


class MertonIteration:
    def __init__(self, 
                 rf_rates:list,
                 ticker:str, 
                 start_date:object, 
                 end_date:object, 
                 disclosure_dates:list,
                 debt_to_equity_ratios:list,
                 maturity:float):
        
        self.rfs = rf_rates
        self.name = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.dis_dates = disclosure_dates 
        self.dtes = debt_to_equity_ratios
        self.T = maturity
        
        self.get_data()
        self.asset_iteration(T=1)
        self.main()
        
        
        
    def get_data(self):
        
        data_stock = yf.download(self.name, start=self.start_date, end=self.end_date)
        data_mkt = yf.download("^GSPC", start=self.start_date, end=self.end_date)
        
        Stock = data_stock['Adj Close']
        SP_500 = data_mkt['Adj Close']
        
        itv1 = np.busday_count(self.dis_dates[0], end_date)
        itv2 = np.busday_count(self.dis_dates[1], end_date)
        itv3 = np.busday_count(self.dis_dates[2], end_date)
        itv4 = np.busday_count(self.dis_dates[3], end_date)
        # itv5 = np.busday_count(self.dis_dates[4], end_date)
        
        lis = []
        for i in range(252 - itv4):
            lis.append(self.dtes[-1])
        for i in range(itv4 - itv3):
            lis.append(self.dtes[-2]) 
        for i in range(itv3 - itv2):
            lis.append(self.dtes[-3]) 
        for i in range(itv2 - itv1):
            lis.append(self.dtes[-4]) 
        for i in range(itv1):
            lis.append(self.dtes[-5]) 
            
        liabilities = np.array(lis) * np.array(Stock)
        
        data = pd.DataFrame([Stock,SP_500]).T
        
        data.columns = ['Equity','sp_500']
        data['Liabilities'] =liabilities
        data['Risk_free'] = rf.to_list()
        self.data = data[['Equity','Liabilities','Risk_free','sp_500']]
        
        df = data.drop('sp_500', axis = 1)
        df['log_risk_free'] = np.log(1+df['Risk_free'])
        self.df = df[['Equity','Liabilities', 'log_risk_free']]
        
        return 0
    
    
    def asset_iteration(self, T=1):
        
        debt = self.df.Liabilities
        equity = self.df.Equity
        rate = self.df.log_risk_free
        book_asset = equity + debt
        
        asset = book_asset.tolist()
        SSE = 1                  
        asset_series = []        
        SSE_series = []          
        asset_vol_series = []    
        asset_vol = 0
        number_iter = 10 
        
        for i in range(number_iter):
            asset_series.append(asset)
            
            if SSE <= 1e-10:
                asset_vol_series.append(asset_vol)
                break
            else:
                
                log_returns = [np.log(asset[i+1]/asset[i]) for i in range(len(asset)-1)]  # Compute asset returns
                asset_vol = np.std(log_returns)*252**0.5    # Compute asset volatility
                
                asset_vol_series.append(asset_vol)
                d1 = (np.log(asset/debt)+(rate + (asset_vol**2)*0.5)*T)/asset_vol*np.sqrt(T)
                d2 = d1 - asset_vol*np.sqrt(T)
                self.df['d1'] = d1     
                self.df['d2'] = d2     # Save d1 and d2 dataframe containing (equity, debt,log_risk_free)

                current_asset = []     # List to hold all the daily asset value for the current iteration
         
                for record in self.df.to_records(index=False): # record = (equity, debt,log_risk_free, d1,d2)
                
                    #Using Black_Scholes_Merton formula to compute currrent asset value in the market
                    equity_,debt_,rate_,d1,d2 = record
                    daily_asset = (equity_ + debt_*exp(-rate_*T)*st.norm.cdf(d2))/st.norm.cdf(d1)
                    
                    current_asset.append(daily_asset)
                    
               
                SSE = np.sum([(i-j)**2  for i,j in zip(asset,current_asset)]) # Compute the sum squared difference
                SSE_series.append(SSE)
                        
                asset = current_asset # Updating for next iteration

        df_aseet_iter = pd.DataFrame(asset_series) # Put all asset iterations into one Dataframe

        asset_iteration_df = df_aseet_iter.T
    
        return asset_iteration_df, asset_vol_series
    
    
    def main(self):
                
        T = self.T
        asset_iteration_df, asset_vol_series = self.asset_iteration(T)
        N = len(asset_iteration_df.columns) - 1
        df_drift = self.data[['Risk_free','sp_500']] 
        df_drift['book_asset'] = asset_iteration_df[0].tolist()
        df_drift['asset_iter'] = asset_iteration_df[N].tolist()
        risk_free_rate = df_drift['Risk_free'].tolist() 
        rf = [i for i in risk_free_rate]
        
        # S&P500
        sp_500 = df_drift['sp_500'].tolist()
        sp_500_ret = [(sp_500[i+1] - sp_500[i])/sp_500[i] for i in range(len(sp_500)-1)] # Compute daily returns for S&P500
        sp_r1 = [i for i in sp_500_ret]
        sp_excess_r1 = [i-j for i,j in zip(sp_r1,rf)]   
        # Compute excess return for S&P500
        V_sp_excess_r1 = [i for i in sp_excess_r1]
        # Fill in nan value in order to match the size of assets in the dataframe below
        V_sp_excess_r1.insert(0,np.nan) 
        # Fill in nan value in order to match the size of S&p500 in the dataframe below
        sp_r1.insert(0,np.nan)   
        
        # Book Asset
        book_asset = df_drift['book_asset'].tolist()
        book_asset_ret = [(book_asset[i+1] - book_asset[i])/book_asset[i] for i in range(len(book_asset)-1)]
        ba_r2 = [i for i in book_asset_ret]
        excess_ba_r2 = [i-j for i,j in zip(ba_r2,rf)]
        V_excess_ba_r2 = [i for i in excess_ba_r2]
        V_excess_ba_r2.insert(0,np.nan)
        ba_r2.insert(0,np.nan)

        # Asset Iterate
        asset_iter = df_drift['asset_iter'].tolist()
        asset_iter_ret = [(asset_iter[i+1] - asset_iter[i])/asset_iter[i] for i in range(len(asset_iter)-1)]
        ait_r2 = [i for i in asset_iter_ret]
        excess_ait_r2 = [i-j for i,j in zip(ait_r2,rf)]
        V_excess_ait_r2 = [i for i in excess_ait_r2]
        V_excess_ait_r2.insert(0,np.nan)
        ait_r2.insert(0,np.nan)
        
        # Dataframe 
        df_drift['sp_500_returns'] = sp_r1
        df_drift['book_asset_ret'] = ba_r2
        df_drift['asset_iter_ret'] = ait_r2
        df_drift['V_sp_excess_r1'] = V_sp_excess_r1
        df_drift['V_excess_ba_r2'] = V_excess_ba_r2
        df_drift['V_excess_ait_r2'] = V_excess_ait_r2
   
        excess_df = pd.DataFrame([sp_excess_r1,excess_ba_r2,excess_ait_r2]).T
        excess_df.columns = ['excess_sp_500', 'excess_ba', 'excess_ait']
        
        # Calculate Beta from book asset and market index (S&P500)
        cov_sp_ba = excess_df[['excess_sp_500','excess_ba']].cov()
        Beta_sp_ba = cov_sp_ba.loc['excess_sp_500','excess_ba']/excess_df[['excess_sp_500']].var()
        Beta_sp_ba = Beta_sp_ba.values[0]
        
        cov_sp_ait = excess_df[['excess_sp_500','excess_ait']].cov()
        Beta_sp_ait = cov_sp_ait.loc['excess_sp_500','excess_ait']/excess_df[['excess_sp_500']].var()
        Beta_sp_ait = Beta_sp_ait.values[0]
        
        # We convert the average daily return to average anualised daily returns:
        E_Rm = (np.mean(sp_500_ret) + 1)**252 - 1

        # average annualised market risk premium is assumed :
        market_primium = E_Rm - rf[-1]  
        
        #Expected return on book asset
        expected_return_1 =  rf[-1] + (Beta_sp_ba * market_primium) 
        #Expected return on asset iterate
        expected_return_2 =  rf[-1] + (Beta_sp_ait*market_primium)
        
        
        
        # drift rate of book asset
        drift_1 = np.log(1+expected_return_1 )
        # drift rate of asset iterate
        drift_2 = np.log(1+expected_return_2 )
        
        capm_variable = {'At maturity 2020-02-06':['book asset', 'asset iterate'],
                 'Beta':[Beta_sp_ba,Beta_sp_ait],
                 'Expected return':[expected_return_1,expected_return_2],
                 'Drift rate': [drift_1,drift_2]}
        haha_df = pd.DataFrame(capm_variable)
        
        debt = self.df.Liabilities
        debt_last = debt[251]   # Debt at maturity

        # Book asset
        asset_1 = asset_iteration_df[0][251]  #Book asset at maturity 
        asset_1_vol = asset_vol_series[0]    #Book_asset's volatility at maturity 
        mu_1 = drift_1    #Drift rate at maturity

        # Probability of default for Book asset
        PD = st.norm.cdf(-((np.log(asset_1/debt_last) + (mu_1 - (asset_1_vol**2)*0.5)*T)/asset_1_vol*np.sqrt(T)))

        # Asset iterate
        asset_iter = asset_iteration_df[N][251]   # Asset_iterate at maturity
        asset_iter_vol = asset_vol_series[-1]    # Asset_iterate's volatility at maturity
        mu_2 = drift_2   # Drift rate at maturity

        #probability of default for Asset iterate
        PD_iter = st.norm.cdf(-((np.log(asset_iter/debt_last) + (mu_1 - (asset_iter_vol**2)*0.5)*T)/asset_iter_vol*np.sqrt(T)))

        PD_variables = {'At maturity 2020-02-06' :['Debt ($L_{T}$)', 
                                           'Asset ($A_{T}$)',
                                           'Volatility ($\sigma_{A}$)', 
                                           'Drift rate ($\mu_{A}$)',
                                           ' ',
                                           'Probability of default ($PD$)'
                                          ],
                'Book asset':[debt_last, asset_1, str(round(asset_1_vol,2)) +'%', str(round(mu_1*100,2)) + '%', ' ', str(round(PD*100,2)) + '%'],
                'Asset iterate':[debt_last, asset_iter, str(round(asset_iter_vol,2)) +'%', str(round(mu_2*100,2)) +'%',' ', str(round(PD_iter*100,2)) + '%']}
                                                            
        kks_df = pd.DataFrame(PD_variables)
        
        print(kks_df)

        self.prob_1y = 1- float(PD_iter)
        self.prob_bk = 1- float(PD)
                
        return 1- float(PD_iter)
    
    
class CdsPricing:
     
    def __init__(self, spreads, spread_tenors, recovery_rate, 
                 risk_free_rate, premium_frequency):
        
        self.sprs = spreads # prices of spreads given in the chart
        self.sptnrs = spread_tenors # tenors of prices of spreads
        self.rc = recovery_rate # as is 
        self.rf = risk_free_rate # as is 
        self.freq = premium_frequency 
        # number of premium which are paid evenly within one year 
        # i.e. premium_frequency = 2 if premiums are paid semi-annually
        self.hazard_rates() # we make sure that once a class object is initialized 
        # we automatically have the values for hazard rates
        self.survival_probs() # we also make sure that survival probs are obtained automatically 
        self.q1()
            
    def q1(self):
        
        self.prob_1y = self.survl_probs[-1]
        
        print('the market implied 1y survival probability =', round(self.survl_probs[-1], 4))
        
        return 0
        
    def premium_leg(self, spread, N, dntrs, intervals, hazard_rates):
        """
        :param spread: corresponding spread, constant for one specific CDS contract.
        :param dntrs: corresponding denominator, list, length = N.
        :param N: the number of premium payments, constant for one specific CDS contract.
        :param intervals: corresponding interval, list, length = N.
        :param hazard_rates : hazard rates, list, length is piece_wise constant, piece-wise constant
        :return: the value of the premium leg
        """
        premiums = 0
        # this in the end returns the value of the premium legs

        prob_begin = 1.0
        # at inception, the survival probability is 1.
        # prb_begin: the probability of not default till the beginning of a specific interval
        # pro_end: the probability of not default till the beginning of a specific interval
        
        for n in range(1, N + 1, 1):
            prob_tail = prob_begin * math.exp(-1 * hazard_rates[n - 1] * intervals[n - 1])
            premiums += 0.5 * spread * pow(dntrs[n - 1], n) * intervals[n - 1] * (prob_begin + prob_tail)
            prob_begin = prob_tail

        # the length of hazard rates should be the same as that of the interval

        return premiums

    def protection_leg(self, N, dntrs, intervals, hazard_rates):
        """
        :param recovery_rate: recovery rate
        :param dntrs: corresponding denominator, list, length = N
        :param N: the number of premium payments, constant for one specific CDS contract.
        :param intervals: corresponding interval, list, length = N
        :param hazard_rates : hazard rates, length = N, piece-wise constant
        :return: the value of the protection leg
        """
        
        recovery_rate = self.rc
        protection = 0
        prob_begin = 1.0
        
        # print('what is the input N?', N)
        
        for n in range(1, N + 1, 1):
            # print('what is n? = ', n)
            prob_tail = prob_begin * math.exp(-1 * hazard_rates[n - 1] * intervals[n - 1])
            
            # print('what is n being here?', n)
            protection += (1 - recovery_rate) * pow(dntrs[n - 1], n) * (prob_begin - prob_tail)
            prob_begin = prob_tail

        return protection
    
    def list_diff(wl1):
        
        wl2 = [0] + wl1[:-1]
        
        resu = [a - b for a,b in zip(wl1, wl2)]
            
        return resu
    
    
    def bisection(self, intervals, dntrs, spread, ind_iter, 
                  xi_resvl, intvls, precision=0.0000000001):
        # bisection method 
        x_inf = 0.0  # hazard rate trials starting points
        x_sup = 1.0
        x1 = 0.25
        d = 1
        hr = xi_resvl + [x1] * int(intvls[ind_iter])
        N_1y = len(hr)
        # rr = self.rc
        
        while abs(d) > precision:
            
            # print('iteration monitor: def bisection')

            # now the trick is that since we can not have hr as an input to 
            # we have to diliver the number of variables in to the function 
            
            d = self.protection_leg(N_1y, dntrs, intervals, hr) - \
                self.premium_leg(spread, N_1y, dntrs, intervals, hr)
                
            # print('protection leg:', self.protection_leg(N_1y, dntrs, intervals, hr))
            # print('premium leg:',  self.premium_leg(spread, N_1y, dntrs, intervals, hr))
    
            if d > 0:
                x_sup = x1
                x1 = x_inf * 0.5 + x1 * 0.5
                # if the protection leg is over-priced, it means that the hazard rates are over-priced
            else:
                x_inf = x1
                x1 = x_sup * 0.5 + x1 * 0.5
                # if the protection leg is under-priced, it means that the hazard rates are under-priced
                
            hr = xi_resvl + [x1] * int(intvls[ind_iter])
            
            # print('x1 =', x1)
            
        return x1
        
    def hazard_rates(self):
        
        xi_resvl = []
        wl1 = self.sptnrs
        wl2 = [0] + wl1[:-1]
        krsu1 = [a - b for a,b in zip(wl1, wl2)]
        intvls = [a * 1.00 * self.freq for a in krsu1]
        
        # my new idea is that we just let denominators be denominators 
        # and switching to the continuously compounded is such a great idea
        # it makes calculation more tractable
        
        for ind_iter in range(len(self.sprs)):
            # print('iteration monitor: repeat at def hazard_rates')
            
            intervals_haha = [1.00/self.freq] * int(sum(intvls[:ind_iter+1]))
            # print('intervals_haha = ', intervals_haha)
            dntrs_haha = [np.exp(self.rf * -0.5)] * int(sum(intvls[:ind_iter+1]))
            # print('dntrs_haha =', dntrs_haha)
            # the discount facotrs are composed this way so that 
            # it incoporates different kinds of yield curve
            spread = self.sprs[ind_iter]
            # print('spread = ', spread)
            
            new_xi = self.bisection(intervals_haha, dntrs_haha, spread,
                                       ind_iter, xi_resvl, intvls)
            
            xi_resvl += [new_xi] * int(intvls[ind_iter])
            
        self.haz_rates = xi_resvl
        # print('baby we get the hazard rates', self.haz_rates)
        
        return 0
    
    def list_product(self, wl):
        
        wll = []
        for i in range(len(wl)):
            ele = 1
            for j in range(i+1):
                ele *= wl[j]
            
            wll.append(ele)
        
        return wll
            
            
    def survival_probs(self):
        
        resu_wl = []
        
        for element in self.haz_rates:
            resu_wl += [math.exp(-0.5 * element)]
            
        self.survl_probs = self.list_product(resu_wl)
        
        return 0
                
            
if __name__ == '__main__':
    
    warnings.filterwarnings('ignore')
    
    spreads_df = pd.read_excel('CDS_spreads.xlsx')
    
    print('Implied from market quotations')
     
    
    start_date = dt.date(2022, 11, 21)
    end_date = dt.date(2023, 11, 22)
    
    dt1 = dt.date(2023, 9, 30)
    dt2 = dt.date(2023, 6, 30)
    dt3 = dt.date(2023, 3, 31)
    dt4 = dt.date(2022, 12, 31)
    dt5 = dt.date(2022, 9, 30)
    dis_dates = [dt1, dt2, dt3, dt4, dt5]
      
    rf = pd.read_csv('risk_free.csv')
    rf = rf.tail(252)
    rf = rf.RF
    rf_rates = rf.to_list()
    dtes_df = pd.read_excel('debt_to_equity_ratio.xlsx')
    Merton_SP = []
    Book_SP = []
    Implied_SP = []
    
    
    for ele in range(5):
        
        ticker = dtes_df.columns[ele+1]
        spreads = np.array(spreads_df.iloc[:, ele+1].to_list()) * (1/ 10000)
        spread_tenors = [0.5, 1, 2, 3, 4, 5, 7, 10]
        rf_value = 0.05
        pf = 2
        rr = 0.4
        print(ticker+':')
        SP = CdsPricing(spreads, spread_tenors, rr,  rf_value, pf)
        Implied_SP.append(SP.prob_1y)
    
    for i in range(5):
        
        print('next')
        dtes = dtes_df.iloc[:, i+1].to_list()
        ticker = dtes_df.columns[i+1]
        
        # survival prob
        SP = MertonIteration(rf_rates, ticker, start_date, end_date, dis_dates, dtes, 1)
        Merton_SP.append(SP.prob_1y)
        Book_SP.append(SP.prob_bk)
        
  
    
    tickers = ("F", "GS", "IBM", "MMM", "BAC")
    PD_measures = {
        'Merton(Book)':rounding(Book_SP),
        'Merton (Iteration)': rounding(Merton_SP),
        'Implied': rounding(Implied_SP)
        
    }

    x = np.arange(len(tickers))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in PD_measures.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Survival Probability')
    ax.set_xlabel('NYSE Listed Companies')
    ax.set_title('1y Survive Probabilities of various companies')
    ax.set_xticks(x + width, tickers)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)
    fig.set_size_inches(16, 9)
    fig.savefig('test.png', dpi=100)

    plt.show()
    
    
    
    
    
    
    
    
        
        
        
        
        
    
        
        
        
        
    
    


