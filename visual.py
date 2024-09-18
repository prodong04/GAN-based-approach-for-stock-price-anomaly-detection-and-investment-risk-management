import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
data = pd.read_csv('bitdata.csv')
anomaly_scores = pd.read_csv('anomaly_score.csv', header=None)

# 날짜 컬럼이 없다면 인덱스를 날짜로 가정하거나 날짜 컬럼을 생성
data['Date'] = pd.to_datetime(data[data.columns[0]])
data = data.sort_values('Date').reset_index(drop=True)

data = data[len(data) - len(anomaly_scores):].reset_index(drop=True)

# anomaly_scores에 날짜가 없다면 데이터 길이에 맞게 날짜 생성
anomaly_scores['Date'] = data['Date'][-len(anomaly_scores):].reset_index(drop=True)
#breakpoint()
# 데이터 병합

# 이동 평균 계산
data['MA30'] = anomaly_scores[1].rolling(window=30).mean()
data['MA60'] = anomaly_scores[1].rolling(window=60).mean()

#breakpoint()
# 신호 생성 함수 정의
def generate_signal(row):
    if np.isnan(row['MA30']) or np.isnan(row['MA60']):
        return 'buy'
    elif row['MA30'] > row['MA60']:
        return 'sell'
    elif row['MA30'] < row['MA60']:
        return 'buy'
    else:
        return 'hold'

# 신호 생성
data['Signal'] = data.apply(generate_signal, axis=1)

# Wallet 클래스 정의
class Wallet:
    def __init__(self, base_currency_name: str, stock_name: str, initial_money: float):
        self.base_currency_name = base_currency_name
        self.stock_name = stock_name
        self.initial_money = initial_money
        self.info = {
            base_currency_name: initial_money,  # 현금 보유량
            stock_name: 0,                      # 주식 보유량
            "buy_count": 0,
            "hold_count": 0,
            "sell_count": 0
        }
        self.total_value = initial_money
        self.profit_percentage = 0

    def buy(self, stock_price: float):
        if self.info[self.base_currency_name] == 0:
            return
        self.info["buy_count"] += 1
        cash = self.info[self.base_currency_name]
        stock_amount = cash / stock_price
        self.info[self.stock_name] = stock_amount
        self.info[self.base_currency_name] = 0

    def hold(self):
        self.info["hold_count"] += 1
        # 보유 상태에서는 별도의 행동 필요 없음

    def sell(self, stock_price: float):
        if self.info[self.stock_name] == 0:
            return
        self.info["sell_count"] += 1
        stock_amount = self.info[self.stock_name]
        cash = stock_amount * stock_price
        self.info[self.base_currency_name] = cash
        self.info[self.stock_name] = 0

    def update_values(self, stock_price: float):
        # 총 자산 가치 계산
        stock_value = self.info[self.stock_name] * stock_price
        cash_value = self.info[self.base_currency_name]
        self.total_value = cash_value + stock_value
        self.profit_percentage = self.total_value / self.initial_money - 1

    def print_values(self):
        print(f"Final portfolio: {self.info}")
        print(f"Total portfolio value: {self.total_value}")
        print(f"Profit percentage: {self.profit_percentage * 100:.2f}%")

# 거래 시뮬레이션
wallet = Wallet("USD", "BTC", 10000)
daily_money = []
dates = []

for idx, row in data.iterrows():
    signal = row['Signal']
    price = row['Close']
    date = row['Date']
    if signal == 'buy':
        wallet.buy(price)
    elif signal == 'sell':
        wallet.sell(price)
    else:
        wallet.hold()
    wallet.update_values(price)
    daily_money.append(wallet.total_value)
    dates.append(date)

wallet.print_values()

# 누적 수익률 계산
daily_money = np.array(daily_money)
daily_money_normalized = daily_money / daily_money[0]

daily_return = daily_money_normalized[1:] / daily_money_normalized[:-1] - 1
cumulative_return = daily_money_normalized[-1] - 1
print(f"Cumulative return: {cumulative_return * 100:.2f}%")

vol = np.std(daily_return) * np.sqrt(252)  # 연간화된 변동성 (거래일 기준)
print(f"Volatility: {vol * 100:.2f}%")

sharpe = (cumulative_return) / vol if vol != 0 else 0
print(f"Sharpe ratio: {sharpe:.2f}")
#breakpoint()
# 성과 시각화
plt.figure(figsize=(12, 6))
plt.plot(dates, daily_money_normalized, label='Strategy Performance')
plt.plot(dates, data['MA30'] , label='MA30 (Normalized)', alpha=0.7)
plt.plot(dates, data['MA60'], label='MA60 (Normalized)', alpha=0.7)
plt.plot(dates, anomaly_scores[1], label='Anomaly Score (Normalized)', alpha=0.7)
plt.plot(dates, data['Close'] / data['Close'].iloc[0], label='BTC Price (Normalized)', alpha=0.7)
plt.title("Strategy Performance vs. BTC Price")
plt.xlabel("Date")
plt.ylabel("Normalized Value")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
