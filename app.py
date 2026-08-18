# -*- coding: utf-8 -*-

import requests
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Nobitex AI Trader",
    layout="wide"
)

st.title("🤖 Nobitex AI Trader")
st.write("اسکن بازار نوبیتکس و نمایش ارزهای دارای روند صعودی")

API = "https://api.nobitex.ir/market/stats"

@st.cache_data(ttl=60)
def get_market():
    try:
        response = requests.get(API, timeout=10)
        data = response.json()

        stats = data.get("stats", {})

        rows = []

        for symbol, item in stats.items():
            if not symbol.endswith("usdt"):
                continue

            try:
                price = float(item.get("latest", 0))
                day_change = float(item.get("dayChange", 0))

                rows.append({
                    "ارز": symbol.upper(),
                    "قیمت": price,
                    "تغییر ۲۴ ساعت": day_change
                })
            except:
                pass

        return pd.DataFrame(rows)

    except Exception as e:
        st.error(f"خطا در دریافت اطلاعات: {e}")
        return pd.DataFrame()


df = get_market()

if df.empty:
    st.warning("اطلاعات بازار دریافت نشد.")
else:

    df = df.sort_values(
        "تغییر ۲۴ ساعت",
        ascending=False
    )

    st.subheader("🔥 ارزهای دارای بیشترین رشد")

    top = df.head(20).copy()

    top["سیگنال"] = top["تغییر ۲۴ ساعت"].apply(
        lambda x: "🟢 صعودی" if x > 3 else
        ("🟡 مثبت" if x > 0 else "🔴 نزولی")
    )

    st.dataframe(
        top,
        use_container_width=True
    )

    st.subheader("🎯 بهترین گزینه‌های بررسی")

    for _, row in top.head(10).iterrows():

        if row["تغییر ۲۴ ساعت"] > 3:

            st.success(
                f"🟢 {row['ارز']} | "
                f"رشد ۲۴ساعته: {row['تغییر ۲۴ ساعت']:.2f}% | "
                f"قیمت: {row['قیمت']}"
            )

st.caption(
    "این برنامه فقط تحلیل و نمایش اطلاعات بازار است و سفارش خرید یا فروش ثبت نمی‌کند."
)
