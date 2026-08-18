# -*- coding: utf-8 -*-
"""
Nobitex AI Scanner v2
اسکن بازارهای ریالی نوبیتکس + امتیازدهی روند + چارت و نقاط ورود/خروج.
فقط از API عمومی استفاده می‌کند و سفارش واقعی ثبت نمی‌کند.
"""
import time, requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

BASE = "https://api.nobitex.ir"
TIMEOUT = 12

st.set_page_config(page_title="Nobitex AI Scanner v2", layout="wide")
st.title("🤖 Nobitex AI Scanner v2")
st.caption("اسکن خودکار بازار ریالی؛ هدف: پیدا کردن شروع روند، نه دنبال‌کردن ارزهای اشباع‌شده.")

@st.cache_data(ttl=30)
def get_stats():
    r = requests.get(f"{BASE}/market/stats", timeout=TIMEOUT)
    r.raise_for_status()
    d = r.json()
    if d.get("status") != "ok":
        raise RuntimeError(d)
    return d.get("stats", {})

@st.cache_data(ttl=30)
def get_candles(symbol, resolution="240", count=180):
    now = int(time.time())
    r = requests.get(
        f"{BASE}/market/udf/history",
        params={"symbol":symbol, "resolution":resolution, "to":now, "countback":count},
        timeout=TIMEOUT
    )
    r.raise_for_status()
    d = r.json()
    if d.get("s") != "ok":
        raise RuntimeError(d)
    return pd.DataFrame({
        "time":pd.to_datetime(d["t"], unit="s"),
        "open":pd.to_numeric(d["o"]),
        "high":pd.to_numeric(d["h"]),
        "low":pd.to_numeric(d["l"]),
        "close":pd.to_numeric(d["c"]),
        "volume":pd.to_numeric(d["v"])
    }).drop_duplicates("time").sort_values("time").reset_index(drop=True)

def ind(df):
    x=df.copy()
    x["ema9"]=x.close.ewm(span=9,adjust=False).mean()
    x["ema20"]=x.close.ewm(span=20,adjust=False).mean()
    x["ema50"]=x.close.ewm(span=50,adjust=False).mean()
    x["ema200"]=x.close.ewm(span=200,adjust=False).mean()
    d=x.close.diff()
    gain=d.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    loss=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rs=gain/loss.replace(0,np.nan)
    x["rsi"]=100-(100/(1+rs))
    e12=x.close.ewm(span=12,adjust=False).mean()
    e26=x.close.ewm(span=26,adjust=False).mean()
    x["macd"]=e12-e26
    x["macds"]=x.macd.ewm(span=9,adjust=False).mean()
    x["hist"]=x.macd-x.macds
    pc=x.close.shift(1)
    tr=pd.concat([x.high-x.low,(x.high-pc).abs(),(x.low-pc).abs()],axis=1).max(axis=1)
    x["atr"]=tr.ewm(alpha=1/14,adjust=False).mean()
    x["vma20"]=x.volume.rolling(20).mean()
    x["high20"]=x.high.rolling(20).max()
    return x

def score_symbol(symbol, stats_row):
    try:
        x=ind(get_candles(symbol))
        if len(x)<80: return None
        a=x.iloc[-1]; p=x.iloc[-2]
        score=0
        reasons=[]

        # 1) fresh-ish price action: positive but not explosive daily move
        dc=float(stats_row.get("dayChange",0))
        if 1 <= dc <= 8:
            score += 18; reasons.append("رشد روزانه مثبت اما هنوز انفجاری نیست")
        elif 0 < dc < 1:
            score += 8; reasons.append("شروع حرکت روزانه")
        elif dc > 12:
            score -= 12; reasons.append("رشد روزانه زیاد؛ خطر تعقیب قیمت")
        elif dc < -5:
            score -= 10

        # 2) EMA structure
        if a.ema9 > a.ema20 > a.ema50:
            score += 22; reasons.append("EMA9>20>50؛ ساختار صعودی")
        elif a.ema20 > a.ema50:
            score += 12; reasons.append("EMA20 بالای EMA50")
        else:
            score -= 15

        # 3) fresh cross
        if p.ema9 <= p.ema20 and a.ema9 > a.ema20:
            score += 15; reasons.append("تقاطع تازه EMA9/20")
        # 4) RSI
        if 45 <= a.rsi <= 65:
            score += 15; reasons.append("RSI مناسب و غیراشباع")
        elif 65 < a.rsi <= 70:
            score += 6; reasons.append("RSI بالا؛ احتیاط")
        elif a.rsi > 70:
            score -= 18; reasons.append("اشباع خرید")
        elif a.rsi < 35:
            score += 3; reasons.append("اشباع فروش؛ نیازمند برگشت")

        # 5) MACD
        if a.hist > 0 and a.hist >= p.hist:
            score += 12; reasons.append("MACD رو به تقویت")
        elif a.hist > 0:
            score += 6
        else:
            score -= 8

        # 6) volume
        if pd.notna(a.vma20) and a.volume > 1.5*a.vma20:
            score += 13; reasons.append("جهش حجم")
        elif pd.notna(a.vma20) and a.volume > a.vma20:
            score += 6

        # 7) avoid being too far from EMA20
        dist=(a.close/a.ema20-1)*100
        if 0 <= dist <= 4:
            score += 10; reasons.append("قیمت نزدیک EMA20")
        elif dist > 8:
            score -= 10; reasons.append("فاصله زیاد از EMA20")

        # Risk levels
        price=float(a.close)
        atr=float(a.atr)
        low10=float(x.low.tail(10).min())
        stop=min(low10, price-1.2*atr)
        risk=max(price-stop, 0.6*atr)
        stop=price-risk
        tp1=price+1.5*risk
        tp2=price+2.2*risk
        tp3=price+3.0*risk

        label = "🟢 ورود مشروط" if score>=72 else ("🟡 تحت نظر" if score>=55 else "⚪ ضعیف")
        return dict(symbol=symbol, score=int(max(0,min(100,score))),
                    dayChange=dc, price=price, rsi=float(a.rsi),
                    volumeRatio=float(a.volume/a.vma20) if a.vma20 else np.nan,
                    entry=price, stop=stop, tp1=tp1, tp2=tp2, tp3=tp3,
                    label=label, reasons="؛ ".join(reasons))
    except Exception:
        return None

# Sidebar
with st.sidebar:
    st.header("تنظیمات اسکن")
    quote=st.selectbox("بازار",["IRT","USDT"],index=0)
    max_scan=st.slider("تعداد ارز برای تحلیل عمیق",20,100,60,step=10)
    min_change=st.slider("حداقل رشد روزانه %",0.0,5.0,0.0,0.5)
    only_fresh=st.checkbox("فقط نامزدهای شروع روند",True)
    auto=st.checkbox("بروزرسانی خودکار هر 60 ثانیه",False)

if auto:
    st.info("برای بروزرسانی خودکار، صفحه را هر 60 ثانیه دوباره بارگذاری کنید.")

try:
    stats=get_stats()
except Exception as e:
    st.error("دریافت بازار انجام نشد.")
    st.code(str(e))
    st.stop()

suffix=quote
rows=[]
for sym,v in stats.items():
    if not sym.endswith(suffix): continue
    if sym.startswith("USDT") and quote=="IRT": continue
    if v.get("isClosed"): continue
    try:
        ch=float(v.get("dayChange",0))
        vol=float(v.get("volumeDst",0))
        latest=float(v.get("latest",0))
    except: continue
    if latest<=0 or ch < min_change: continue
    rows.append((sym,v,ch,vol))
rows=sorted(rows,key=lambda z:z[3],reverse=True)[:max_scan]

st.write(f"بازارهای کاندید: **{len(rows)}**")
progress=st.progress(0)
results=[]
for i,(sym,v,ch,vol) in enumerate(rows):
    z=score_symbol(sym,v)
    if z: results.append(z)
    progress.progress((i+1)/max(1,len(rows)))
progress.empty()

if not results:
    st.warning("نامزد قابل تحلیل پیدا نشد. فیلترها را کمی بازتر کن.")
    st.stop()

df=pd.DataFrame(results).sort_values(["score","dayChange"],ascending=False)
if only_fresh:
    df=df[(df.score>=55)&(df.rsi<70)].copy()

st.subheader("🏆 بهترین نامزدهای فعلی")
show=df[["symbol","score","label","dayChange","rsi","volumeRatio","entry","stop","tp1","tp2","tp3"]].copy()
show.columns=["نماد","امتیاز","وضعیت","تغییر روزانه %","RSI14","نسبت حجم","ورود","حد ضرر","TP1","TP2","TP3"]
st.dataframe(show.head(10),use_container_width=True,hide_index=True)

if len(df):
    selected=st.selectbox("برای دیدن چارت انتخاب کن",df.symbol.head(10).tolist())
    z=df[df.symbol==selected].iloc[0]
    x=ind(get_candles(selected))
    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=x.time,open=x.open,high=x.high,low=x.low,close=x.close,name=selected))
    fig.add_trace(go.Scatter(x=x.time,y=x.ema20,name="EMA20"))
    fig.add_trace(go.Scatter(x=x.time,y=x.ema50,name="EMA50"))
    for y,n,d in [(z.entry,"ENTRY","solid"),(z.stop,"STOP","dash"),(z.tp1,"TP1","dot"),(z.tp2,"TP2","dot"),(z.tp3,"TP3","dot")]:
        fig.add_hline(y=y,line_dash=d,annotation_text=n)
    fig.update_layout(height=650,xaxis_rangeslider_visible=False)
    st.plotly_chart(fig,use_container_width=True)
    a,b,c,d=st.columns(4)
    a.metric("سیگنال",z.label)
    b.metric("Entry",f"{z.entry:,.0f}")
    c.metric("Stop",f"{z.stop:,.0f}")
    d.metric("TP1 / TP2",f"{z.tp1:,.0f} / {z.tp2:,.0f}")
    st.write("**دلایل:**",z.reasons)

st.warning("این موتور سیگنال قطعی یا تضمین سود نیست. قبل از معامله واقعی، با حجم کم/حساب آزمایشی و بک‌تست بررسی شود. این نسخه سفارش واقعی ثبت نمی‌کند.")
