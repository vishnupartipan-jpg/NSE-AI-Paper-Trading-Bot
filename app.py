import sqlite3
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

DB=Path(__file__).with_name("paper_trader_v5.sqlite3")
START=100000.0
st.set_page_config(page_title="NSE AI Trader V5",page_icon="📈",layout="wide")

def conn(): return sqlite3.connect(DB,check_same_thread=False)
with conn() as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY,ts TEXT,symbol TEXT,side TEXT,qty INTEGER,price REAL,fee REAL);
    CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY,ts TEXT,symbol TEXT,level TEXT,message TEXT,ack INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value REAL);
    """)
    defaults={"max_position_pct":20.0,"daily_loss_pct":3.0,"fee_pct":0.05}
    for k,v in defaults.items(): c.execute("INSERT OR IGNORE INTO settings VALUES(?,?)",(k,v))

def settings():
    with conn() as c: return dict(c.execute("SELECT key,value FROM settings").fetchall())
def save_settings(d):
    with conn() as c:
        for k,v in d.items(): c.execute("INSERT OR REPLACE INTO settings VALUES(?,?)",(k,float(v)))
def symbolize(s):
    s=s.strip().upper().replace(" ","")
    return s if not s or s.endswith(".NS") else s+".NS"

@st.cache_data(ttl=300)
def prices(sym,period):
    try:
        d=yf.download(sym,period=period,interval="1d",auto_adjust=True,progress=False)
        if d.empty:return pd.DataFrame()
        if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
        d.columns=[str(x).title() for x in d.columns]
        d=d[["Open","High","Low","Close","Volume"]].dropna().copy()
        c=d.Close
        d["EMA20"]=c.ewm(span=20,adjust=False).mean()
        d["EMA50"]=c.ewm(span=50,adjust=False).mean()
        delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
        loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
        d["RSI"]=100-100/(1+gain/loss.replace(0,np.nan))
        d["MACD"]=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
        d["MACDsig"]=d.MACD.ewm(span=9,adjust=False).mean()
        return d
    except Exception:return pd.DataFrame()

def signal(d):
    r=d.iloc[-1]; score=0; why=[]
    for ok,yes,no in [(r.EMA20>r.EMA50,"EMA20 above EMA50","EMA20 below EMA50"),
                      (r.MACD>r.MACDsig,"MACD above signal","MACD below signal")]:
        score += 1 if ok else -1; why.append(yes if ok else no)
    if pd.notna(r.RSI):
        if r.RSI>55: score+=1; why.append(f"RSI positive ({r.RSI:.1f})")
        elif r.RSI<45: score-=1; why.append(f"RSI weak ({r.RSI:.1f})")
        else: why.append(f"RSI neutral ({r.RSI:.1f})")
    return ("BUY" if score>=2 else "SELL" if score<=-2 else "HOLD"),score,why

def log_alert(s,level,msg):
    with conn() as c:c.execute("INSERT INTO alerts(ts,symbol,level,message) VALUES(?,?,?,?)",(datetime.now().isoformat(timespec="seconds"),s,level,msg))

def ledger():
    with conn() as c:return pd.read_sql_query("SELECT * FROM trades ORDER BY id",c)

def portfolio(t):
    cash=START; pos={}; realized=0.0
    for r in t.itertuples():
        p=pos.setdefault(r.symbol,{"qty":0,"cost":0.0})
        if r.side=="BUY":
            cash-=r.qty*r.price+r.fee;p["qty"]+=r.qty;p["cost"]+=r.qty*r.price+r.fee
        else:
            avg=p["cost"]/p["qty"] if p["qty"] else 0
            realized+=(r.price-avg)*r.qty-r.fee
            p["cost"]-=avg*r.qty;p["qty"]-=r.qty;cash+=r.qty*r.price-r.fee
    return cash,{s:p for s,p in pos.items() if p["qty"]>0},realized

def backtest(d,fee):
    x=d.dropna().copy()
    if len(x)<60:return None
    sc=np.where(x.EMA20>x.EMA50,1,-1)+np.where(x.RSI>55,1,np.where(x.RSI<45,-1,0))+np.where(x.MACD>x.MACDsig,1,-1)
    pos=pd.Series(np.where(sc>=2,1,np.where(sc<=-2,-1,0)),index=x.index).shift(1).fillna(0)
    ret=x.Close.pct_change().fillna(0); turnover=pos.diff().abs().fillna(pos.abs())
    strat=pos*ret-turnover*fee/100
    eq=(1+strat).cumprod(); bh=(1+ret).cumprod()
    dd=eq/eq.cummax()-1
    return eq,bh,(eq.iloc[-1]-1)*100,(bh.iloc[-1]-1)*100,dd.min()*100

st.title("📈 NSE AI Trader V5")
st.caption("Mobile-friendly stock analysis and paper trading. No real orders or broker connection.")
with st.sidebar:
    st.header("Market data")
    raw=st.text_input("NSE ticker","RELIANCE")
    sym=symbolize(raw)
    period=st.selectbox("History",["3mo","6mo","1y","2y","5y"],2)
    if st.button("Refresh data"):prices.clear()
    st.caption("Yahoo Finance data may be delayed, incomplete or unavailable.")
d=prices(sym,period) if sym else pd.DataFrame()
if d.empty or len(d)<55:
    st.error("Insufficient price data. Try RELIANCE, TCS, INFY or another NSE ticker.")
    st.stop()
r=d.iloc[-1]; sig,score,why=signal(d)
st.subheader(f"{sym} · Last available daily bar: {d.index[-1].strftime('%d %b %Y')}")
a,b,c,e=st.columns(4)
a.metric("Last close",f"₹{r.Close:,.2f}");b.metric("RSI 14",f"{r.RSI:.1f}")
c.metric("EMA 20",f"₹{r.EMA20:,.2f}");e.metric("Signal",sig,f"Score {score:+d}/3")
fig=go.Figure()
fig.add_trace(go.Candlestick(x=d.index,open=d.Open,high=d.High,low=d.Low,close=d.Close,name="Price"))
fig.add_trace(go.Scatter(x=d.index,y=d.EMA20,name="EMA 20"))
fig.add_trace(go.Scatter(x=d.index,y=d.EMA50,name="EMA 50"))
fig.update_layout(height=430,xaxis_rangeslider_visible=False,margin=dict(l=5,r=5,t=20,b=5))
st.plotly_chart(fig,use_container_width=True)
with st.expander("Signal explanation",True):
    st.write("Rule-based indicator score, not an AI prediction.")
    for w in why:st.write("• "+w)

st.header("🛡️ Risk controls")
cfg=settings()
with st.form("risk"):
    x,y=st.columns(2)
    mp=x.slider("Max position size (% of starting cash)",5,50,int(cfg["max_position_pct"]),5)
    dl=y.slider("Daily loss alert threshold (%)",1,10,int(cfg["daily_loss_pct"]),1)
    fee=st.number_input("Estimated fee per side (%)",0.0,2.0,float(cfg["fee_pct"]),0.01)
    if st.form_submit_button("Save risk settings"):
        save_settings({"max_position_pct":mp,"daily_loss_pct":dl,"fee_pct":fee});st.success("Saved");st.rerun()

st.header("💼 Paper trading")
t=ledger();cash,positions,realized=portfolio(t)
mark=float(r.Close); current=positions.get(sym,{"qty":0})["qty"]
hold_value=0; unreal=0; rows=[]
for s,p in positions.items():
    q=prices(s,"5d")
    px=float(q.Close.iloc[-1]) if not q.empty else p["cost"]/p["qty"]
    val=p["qty"]*px; hold_value+=val;unreal+=val-p["cost"]
    rows.append({"Symbol":s,"Qty":p["qty"],"Average cost":p["cost"]/p["qty"],"Last price":px,"Market value":val,"Unrealized P&L":val-p["cost"]})
a,b,c,e=st.columns(4)
a.metric("Virtual cash",f"₹{cash:,.2f}");b.metric("Holdings",f"₹{hold_value:,.2f}")
c.metric("Unrealized P&L",f"₹{unreal:,.2f}");e.metric("Realized P&L",f"₹{realized:,.2f}")
st.metric("Total virtual portfolio",f"₹{cash+hold_value:,.2f}",f"{cash+hold_value-START:,.2f} vs starting ₹{START:,.0f}")
buytab,selltab=st.tabs(["Simulate BUY","Simulate SELL"])
fee_rate=cfg["fee_pct"]/100
with buytab:
    with st.form("buy"):
        qty=st.number_input("Shares to buy",1,100000,1)
        st.write(f"Estimated cost: ₹{qty*mark:,.2f} plus estimated fees")
        go=st.form_submit_button("Record paper BUY")
    if go:
        val=qty*mark; cap=START*cfg["max_position_pct"]/100
        if positions.get(sym,{"qty":0})["qty"]*mark+val>cap:st.error(f"Position cap is ₹{cap:,.2f}.")
        elif val*(1+fee_rate)>cash:st.error("Not enough virtual cash.")
        else:
            with conn() as db:db.execute("INSERT INTO trades(ts,symbol,side,qty,price,fee) VALUES(?,?,?,?,?,?)",(datetime.now().isoformat(timespec="seconds"),sym,"BUY",int(qty),mark,val*fee_rate))
            log_alert(sym,"TRADE",f"Paper BUY {qty} shares at ₹{mark:.2f}");st.success("Paper BUY recorded.");st.rerun()
with selltab:
    with st.form("sell"):
        sq=st.number_input("Shares to sell",1,max(1,current),1,disabled=current==0)
        go2=st.form_submit_button("Record paper SELL",disabled=current==0)
    if go2:
        if sq>current:st.error("Insufficient shares.")
        else:
            val=sq*mark
            with conn() as db:db.execute("INSERT INTO trades(ts,symbol,side,qty,price,fee) VALUES(?,?,?,?,?,?)",(datetime.now().isoformat(timespec="seconds"),sym,"SELL",int(sq),mark,val*fee_rate))
            log_alert(sym,"TRADE",f"Paper SELL {sq} shares at ₹{mark:.2f}");st.success("Paper SELL recorded.");st.rerun()
if rows:st.subheader("Open holdings");st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
else:st.info("No open holdings. Use simulated BUY to test the paper account.")
st.subheader("Trade history")
if t.empty:st.write("No trades yet.")
else:st.dataframe(t,use_container_width=True,hide_index=True)

st.header("🧪 Historical strategy backtest")
st.caption("Signals are shifted to the next session. Simplified simulation, not a forecast.")
if st.button("Run backtest"):
    result=backtest(d,cfg["fee_pct"])
    if result is None:st.warning("Select a longer history period.")
    else:
        eq,bh,total,buyhold,maxdd=result
        a,b,c=st.columns(3);a.metric("Strategy return",f"{total:.2f}%");b.metric("Buy & hold",f"{buyhold:.2f}%");c.metric("Max drawdown",f"{maxdd:.2f}%")
        f=go.Figure();f.add_trace(go.Scatter(x=eq.index,y=eq,name="Strategy"));f.add_trace(go.Scatter(x=bh.index,y=bh,name="Buy & hold"))
        f.update_layout(height=300);st.plotly_chart(f,use_container_width=True)
st.header("🔔 In-app alerts")
with conn() as db:alerts=pd.read_sql_query("SELECT * FROM alerts ORDER BY id DESC LIMIT 50",db)
if alerts.empty:st.info("Trade activity alerts will appear here.")
else:
    st.dataframe(alerts,use_container_width=True,hide_index=True)
    if st.button("Acknowledge alerts"):
        with conn() as db:db.execute("UPDATE alerts SET ack=1")
        st.rerun()
with st.expander("Reset paper account"):
    if st.checkbox("I understand this deletes all local simulated trades and alerts"):
        if st.button("Reset account"):
            with conn() as db:db.execute("DELETE FROM trades");db.execute("DELETE FROM alerts")
            st.success("Reset complete.");st.rerun()
st.caption("Educational paper trading only. No broker API, live order execution, SMS/push alerts or guaranteed returns. Stop-loss/take-profit execution, taxes, slippage, market impact, liquidity and corporate actions are not fully modeled. Backtests do not predict future results. Streamlit cloud local storage may not be durable.")
