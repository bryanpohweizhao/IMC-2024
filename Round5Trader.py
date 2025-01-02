from datamodel import OrderDepth, UserId, TradingState, Order
import string
import numpy as np
import math
import copy
import jsonpickle
import collections
from collections import defaultdict
from typing import List, Dict

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
class Trader:
    

    def run(self, state):
        result = {'AMETHYSTS':[], 'STARFRUIT':[], 'GIFT_BASKET':[], 'CHOCOLATE':[], 'STRAWBERRIES':[], 'ROSES':[], 'COCONUT':[], 'COCONUT_COUPON':[]}
        traderData = state.traderData or jsonpickle.encode({})
        past_data = jsonpickle.decode(traderData)
        conversions = 0

        for product in state.order_depths:
            if product == 'STARFRUIT':
                trader = self.StarFruitTrader(state=state, product=product, past_data=past_data) 
                order, past_data = trader.execute()
                result[product] = order
                
            if product == 'AMETHYSTS':
                trader = self.AmethystsTrader(state=state, product=product, past_data=past_data) 
                order, past_data = trader.execute()
                result[product] = order

            if product == 'GIFT_BASKET':
                trader = self.GiftBasketTrader(state=state, past_data=past_data)
                order, past_data = trader.execute()
                for prod, ord in order.items():
                    result[prod] = ord

            if product == 'COCONUT_COUPON':
                trader = self.CoconutTrader(state=state, past_data=past_data)
                order, past_data = trader.execute()
                result[product] = order

            if product == 'ORCHIDS':
                observation = state.observations.conversionObservations[product]
                buy_from_ducks = observation.askPrice + observation.transportFees - observation.importTariff
                sell_to_ducks = observation.bidPrice + observation.transportFees - observation.exportTariff
                midprice_ducks = (buy_from_ducks + sell_to_ducks)/2

                order_depth = state.order_depths[product]
                osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
                obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

                askprice1, _ = next(iter(osell.items()))
                bidprice1, _ = next(iter(obuy.items()))
                midprice = (askprice1 + bidprice1)/2

                cpos = state.position.get(product, 0)
                limit = 100
                orders = []

                long_profit = max(0, sell_to_ducks - askprice1)
                short_profit = max(0, bidprice1 - buy_from_ducks)

                change = 3
                if state.timestamp == 0:
                    past_data['ORCHIDS'] = {'prev_price': midprice_ducks, 'prev_change': change}

                if state.timestamp != 0:
                    change = np.abs(past_data['ORCHIDS']['prev_price'] - midprice_ducks)
                    past_data['ORCHIDS'] = {'prev_price': midprice_ducks, 'prev_change': change}
                
                if (long_profit > 0) and (long_profit > past_data['ORCHIDS']['prev_change']) and (long_profit > short_profit):
                    max_vol = int(math.floor((long_profit - past_data['ORCHIDS']['prev_change'])/0.1))
                    order_vol = min(max_vol, limit-cpos)
                    if order_vol > 0:
                        orders.append(Order(product, askprice1, order_vol))
                
                if (short_profit > 0) and (short_profit > past_data['ORCHIDS']['prev_change']) and (short_profit > long_profit):
                    max_vol = int(math.floor((short_profit - past_data['ORCHIDS']['prev_change'])/0.1))
                    order_vol = min(max_vol, limit+cpos)
                    if order_vol > 0:
                        order.append(Order(product, bidprice1, -order_vol))
                
                conversions = -cpos
                result[product] = orders

        traderData = jsonpickle.encode(past_data)
        return result, conversions, traderData
    
    class CoconutTrader:
        def __init__(self, state, past_data):
            self.option = 'COCONUT_COUPON'
            self.underlying = 'COCONUT'
            self.askprice = {}
            self.bidprice = {}
            self.midprice = {}
            
            for prod in ['COCONUT_COUPON', 'COCONUT']:
                order_depth = state.order_depths[prod]
                osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
                obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

                askprice1 = next(iter(osell.keys()), 0)
                bidprice1 = next(iter(obuy.keys()), 0)
                if bidprice1 == 0:
                    bidprice1 = askprice1
                if askprice1 == 0:
                    askprice1 = bidprice1
                
                self.askprice[prod] = askprice1
                self.bidprice[prod] = bidprice1
                self.midprice[prod] = (askprice1 + bidprice1)/2

            self.state = state
            self.past_data = past_data
            self.spread_std = 13.5

            
        def compute_option_price(self):

            intercept = 627.4890236934032
            deviation = 0.42352123
            option_val = 0.08682443
            itm_ind = 7.79092987
            underlying_price = self.midprice['COCONUT']
            return intercept + deviation*(underlying_price - 10000) + option_val*max(0, underlying_price - 10000) + itm_ind*np.where(underlying_price >= 10000, 1, 0)
        
        def execute(self):
            ## The usual Stuff
            orders = []
            limit = 600
            cpos = self.state.position.get(self.option, 0)

            ## Compute Option Mispricing
            fitted_price = self.compute_option_price()
            option_price = self.midprice[self.option]
            spread = option_price - fitted_price

            ## Store prior mispricing
            if self.state.timestamp == 0:
                self.past_data[self.option] = {'prior_spread' : spread}
            
            ## Short OPTION
            if spread > self.spread_std*0.5:
                order_vol = limit + cpos
                if order_vol > 0:
                    orders.append(Order(self.option, self.bidprice[self.option], -order_vol))

            ## Long OPTION
            if spread < -self.spread_std*0.5:
                order_vol = limit - cpos
                if order_vol > 0:
                    orders.append(Order(self.option, self.askprice[self.option], order_vol))
            
            ## CLOSE SHORT
            if (spread < 0) and (self.past_data[self.option]['prior_spread'] > 0) and (cpos < 0):
                orders.append(Order(self.option, self.askprice[self.option], -cpos))

            ## CLOSE LONG
            if (spread > 0) and (self.past_data[self.option]['prior_spread'] < 0) and (cpos > 0):
                orders.append(Order(self.option, self.bidprice[self.option], -cpos))

            ## UPDATE PAST_DATA
            self.past_data[self.option]['prior_spread'] = spread
            return orders, self.past_data
        
    class GiftBasketTrader:
        def __init__(self, state, past_data):
            self.trader = 'Rhianna'
            self.products = ['CHOCOLATE', 'STRAWBERRIES', 'ROSES', 'GIFT_BASKET']
            self.limit = {'GIFT_BASKET':60, 'ROSES':60}
            self.orders = {'GIFT_BASKET':[], 'ROSES':[]}
            self.askprice = {}
            self.bidprice = {}
            self.midprice = {}
            self.positions = {}

            for prod in self.products:
                order_depth = state.order_depths[prod]
                osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
                obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

                askprice1, _ = next(iter(osell.items()))
                bidprice1, _ = next(iter(obuy.items()))

                self.askprice[prod] = askprice1
                self.bidprice[prod] = bidprice1
                self.midprice[prod] = (askprice1 + bidprice1)/2
                self.positions[prod] = state.position.get(prod, 0)
            
            self.state = state
            self.past_data = past_data

            if state.timestamp == 0:
                past_data['GIFT_BASKET'] = {'cache':[], 'cumm_mean':0}

            self.past_data = past_data
            self.spread_mean = 380
            self.spread_std = 80
            self.dim = 6
               
        def get_price_slope(self, prod):
            x = [-3, -2, -1, 1, 2, 3]
            x = (x-np.mean(x))/np.std(x, ddof=0)

            y = self.past_data[prod]['cache']
            y_std = np.std(y, ddof=0)
            if y_std == 0:
                return 0
            else:
                y = (y - np.mean(y))/y_std

                numerator, denominator = 0, 0
                for i in range(len(x)):
                    numerator += x[i]*y[i]
                    denominator += x[i]**2
                return numerator/denominator
        
        def record_data(self):
            prod = 'GIFT_BASKET'
            data = self.past_data[prod]
            if len(data['cache']) == self.dim:
                data['cache'].pop(0)
            data['cache'].append(self.midprice[prod])


        def execute(self):

            #########################
            ### Trade Gift Basket ###
            #########################
            self.record_data()
            data = self.past_data['GIFT_BASKET']

            # Compute acceptable premium
            premium = self.midprice['GIFT_BASKET'] - (self.midprice['CHOCOLATE']*4 + self.midprice['STRAWBERRIES']*6 + self.midprice['ROSES'])
            time = self.state.timestamp/100
            data['cumm_mean'] = (data['cumm_mean']*max(0, time-1) + premium)/max(1, time)

            # Compute current spread based on current acceptable premium
            adj_spread = ((10000-time)*self.spread_mean + time*data['cumm_mean'])/10000
            spread = premium - adj_spread

            # Set Entry and Exit Params
            trade_at = self.spread_std*0.4
            slope = 0
            if len(data['cache']) == self.dim:
                slope = self.get_price_slope('GIFT_BASKET')
                    
            #SHORT BASKET
            if (spread > trade_at):
                #IF DOWN TREND, SHORT BASKET
                if (slope < 0.25):
                    order_vol = self.limit['GIFT_BASKET'] + self.positions['GIFT_BASKET']
                    if order_vol > 0:
                        self.orders['GIFT_BASKET'].append(Order('GIFT_BASKET', self.bidprice['GIFT_BASKET'], -order_vol))
        
            # LONG BASKET
            if (spread < -trade_at):
                #IF UPTREND, LONG BASKET
                if (slope > -0.25):
                    order_vol = self.limit['GIFT_BASKET'] - self.positions['GIFT_BASKET']
                    if order_vol > 0:
                        self.orders['GIFT_BASKET'].append(Order('GIFT_BASKET', self.askprice['GIFT_BASKET'], order_vol))

            ###################
            ### TRADE ROSES ###
            ###################
            rose = 'ROSES'

            ###############################
            ### Check if Rhianna Traded ###
            ###############################
            if rose in self.state.market_trades.keys():
                rose_market_trades = self.state.market_trades[rose]
                for trade in rose_market_trades:
                    
                    if (trade.buyer == trade.seller) and (trade.buyer == self.trader):
                        continue

                    if trade.buyer == self.trader:
                        order_vol = self.limit[rose] - self.positions[rose]
                        self.orders[rose].append(Order(rose, self.askprice[rose], order_vol))

                    if trade.seller == self.trader:
                        order_vol = self.limit[rose] + self.positions[rose]
                        self.orders[rose].append(Order(rose, self.bidprice[rose], -order_vol))
           
            return self.orders, self.past_data
            
    ## STARFRUITS STRATEGY:
    class StarFruitTrader:
        def __init__(self, state, product, past_data):
            self.limit = 20
            self.coef = [0.3417033,  0.25739856, 0.20912411, 0.19172258]
            self.intercept = 0.2531416221472682
            self.dim = 4
            self.state = state
            self.past_data = past_data
            self.product = product
            if state.timestamp == 0:
                self.past_data[product] = {'cache':[], 'position':0}
            self.past_data[self.product]['position'] = self.state.position.get(self.product, 0)
                
        def compute_next_price(self):
            
            data = self.past_data[self.product]
            nxt_price = self.intercept
            for i, val in enumerate(data['cache']):
                nxt_price += val * self.coef[i]
            return int(round(nxt_price))

        def execute(self):
            # Get Market Data
            orders = []
            order_depth = self.state.order_depths[self.product]
            osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
            obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

            askprice1, _ = next(iter(osell.items()))
            bidprice1, _ = next(iter(obuy.items()))
            midprice = (askprice1 + bidprice1)/2

            data = self.past_data[self.product]
            if len(data['cache']) == self.dim:
                data['cache'].pop(0)
            
            data['cache'].append(midprice)
            INF = int(1e9)
            acc_bid, acc_ask = -INF, INF

            if len(data['cache']) == self.dim:
                acc_bid = self.compute_next_price()-1
                acc_ask = self.compute_next_price()+1
            
            cpos = data['position']
            for ask, vol in osell.items():
                if ((ask <= acc_bid) or ((data['position'] < 0) and (ask == acc_bid+1))) and cpos < self.limit:
                    order_vol = min(-vol, self.limit-cpos)
                    cpos += order_vol
                    assert(order_vol >= 0)
                    orders.append(Order(self.product, ask, order_vol))

            undercut_buy = bidprice1 + 1
            undercut_sell = askprice1 - 1

            buy_price = min(undercut_buy, acc_bid-1)
            sell_price = max(undercut_sell, acc_ask+1)

            if cpos < self.limit:
                num = self.limit - cpos
                orders.append(Order(self.product, buy_price, num))
                cpos += num
            
            cpos = data['position']

            for bid, vol in obuy.items():
                if ((bid >= acc_ask) or ((data['position'] > 0) and (bid+1 == acc_ask))) and cpos > -self.limit:
                    order_vol = max(-vol, -self.limit-cpos)
                    cpos += order_vol
                    assert(order_vol <= 0)
                    orders.append(Order(self.product, bid, order_vol))
            
            if cpos > -self.limit:
                num = -self.limit-cpos
                orders.append(Order(self.product, sell_price, num))
                cpos += num
  
            return orders, self.past_data
        
    # Amethsysts Strategy
    class AmethystsTrader:
        def __init__(self, state, product, past_data):
            self.limit = 20
            self.state = state
            self.past_data = past_data
            self.product = product
            if state.timestamp == 0:
                self.past_data[product] = {'position':0}
            self.past_data[self.product]['position'] = self.state.position.get(self.product, 0)
        
        
        def execute(self):
            orders = []
            order_depth = self.state.order_depths[self.product]
            osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
            obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

            askprice1, _ = next(iter(osell.items()))
            bidprice1, _ = next(iter(obuy.items()))
            acc_bid, acc_ask = 10000, 10000

            data = self.past_data[self.product]
            cpos = data['position']

            mx_with_buy = -1
            for ask, vol in osell.items():
                if ((ask < acc_bid) or ((data['position']<0) and (ask == acc_bid))) and cpos < self.limit:
                    mx_with_buy = max(mx_with_buy, ask)
                    order_vol = min(-vol, self.limit - cpos)
                    cpos += order_vol
                    assert(order_vol >= 0)
                    orders.append(Order(self.product, ask, order_vol))

            undercut_buy = bidprice1 + 1
            undercut_sell = askprice1 - 1
            bid_pr = min(undercut_buy, acc_bid-1)
            sell_pr = max(undercut_sell, acc_ask+1)

            if (cpos < self.limit):
                if (data['position'] < 0):
                    num = min(40, self.limit - cpos)
                    orders.append(Order(self.product, min(undercut_buy + 1, acc_bid-1), num))
                    cpos += num
                if (data['position'] > 15):
                    num = min(40, self.limit - cpos)
                    orders.append(Order(self.product, min(undercut_buy - 1, acc_bid-1), num))
                    cpos += num
            if (cpos < self.limit):
                num = min(40, self.limit - cpos)
                orders.append(Order(self.product, bid_pr, num))
                cpos += num
                
            
            cpos = data['position']
            for bid, vol in obuy.items():
                if ((bid > acc_ask) or ((data['position']>0) and (bid == acc_ask))) and cpos > -self.limit:
                    order_vol = max(-vol, -self.limit-cpos)
                    cpos += order_vol
                    assert(order_vol <= 0)
                    orders.append(Order(self.product, bid, order_vol))

            if (cpos > -self.limit):
                if (data['position'] > 0):
                    num = max(-40, -self.limit-cpos)
                    orders.append(Order(self.product, max(undercut_sell-1, acc_ask+1), num))
                    cpos += num
                if (data['position'] < -15):
                    num = max(-40, -self.limit-cpos)
                    orders.append(Order(self.product, max(undercut_sell+1, acc_ask+1), num))
                    cpos += num

            if cpos > -self.limit:
                num = max(-40, -self.limit-cpos)
                orders.append(Order(self.product, sell_pr, num))
                cpos += num

            return orders, self.past_data
    
    