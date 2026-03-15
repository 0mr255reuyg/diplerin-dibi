import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ══════════════════════════════════════════════════════
# GÜNCEL BIST 100 LİSTESİ (Mart 2025)
# ══════════════════════════════════════════════════════
BIST100 = [
    "ACSEL","ADEL","AEFES","AGESA","AKBNK","AKSA","AKSEN","ALARK","ALBRK",
    "ALFAS","ALKIM","ANACM","ARCLK","ARDYZ","ASELS","ASTOR","BERA","BIMAS",
    "BIOEN","BRISA","BRYAT","BUCIM","CCOLA","CIMSA","CLEBI","CVKMD",
    "DOAS","DOHOL","DSTKF","ECILC","ENJSA","ENKAI","EREGL","EUPWR",
    "FROTO","GARAN","GESAN","GLYHO","GOLTS","GUBRF","HALKB","HEKTS",
    "IPEKE","ISCTR","ISFIN","ISGYO","ISYHO","ITTFK","IZOCM",
    "KAREL","KARSN","KCAER","KCHOL","KLMSN","KONTR","KONYA","KORDS",
    "KOZAA","KOZAL","KRDMD","LOGO","MAVI","MGROS","MPARK","NETAS",
    "ODAS","OTKAR","OYAKC","PETKM","PGSUS","PKENT","POLHO","QUAGR",
    "RGYAS","SAHOL","SASA","SELEC","SISE","SKBNK","SMART","SOKM",
    "TATGD","TCELL","THYAO","TKFEN","TOASO","TSKB","TTKOM","TTRAK",
    "TUPRS","TURSG","ULKER","VAKBN","VESTL","YKBNK","ZOREN",
    "AGHOL","BFREN","CEMTS","DYOBY","PRKME"
]

R = {
    "TKE":"#FF6B35","StochRSI":"#4ECDC4","MFI":"#FFE66D","RSI":"#A8DADC",
    "fiyat":"#F7FFF7","bg":"#1A1A2E","panel":"#16213E","txt":"#E0E0E0",
    "g":"#06D6A0","r":"#EF476F",
}

# ══════════════════════════════════════════════════════
# VERİ ÇEKME
# ══════════════════════════════════════════════════════
def veri_cek(hisse, periyot="1y"):
    try:
        df = yf.Ticker(hisse+".IS").history(period=periyot, interval="1d", auto_adjust=True)
        if df is None or df.empty or len(df) < 50:
            return None
        return df[["Open","High","Low","Close","Volume"]].copy()
    except:
        return None

# ══════════════════════════════════════════════════════
# İNDİKATÖRLER
# ══════════════════════════════════════════════════════
def hesapla_tke(df, period=21):
    h,l,c,v = df["High"],df["Low"],df["Close"],df["Volume"]
    hlc3 = (h+l+c)/3
    mom = c/c.shift(period)*100
    cci = (hlc3-hlc3.rolling(period).mean())/(0.015*hlc3.rolling(period).std())
    dlt = c.diff()
    g = dlt.clip(lower=0).rolling(period).mean()
    ls = (-dlt.clip(upper=0)).rolling(period).mean()
    rsi = 100-100/(1+g/ls.replace(0,np.nan))
    hi = h.rolling(period).max(); lo = l.rolling(period).min()
    willr = (hi-c)/(hi-lo+1e-10)*-100
    stosk = (c-lo)/(hi-lo+1e-10)*100
    dif = hlc3.diff()
    up = (v*hlc3.where(dif>0,0)).rolling(period).sum()
    dn = (v*hlc3.where(dif<0,0)).rolling(period).sum()
    mfi = 100-100/(1+up/dn.replace(0,np.nan))
    h_ = pd.concat([h,c.shift(1)],axis=1).max(axis=1)
    l_ = pd.concat([l,c.shift(1)],axis=1).min(axis=1)
    bp = c-l_; tr_ = h_-l_
    def avg(b,t,n): return b.rolling(n).sum()/t.rolling(n).sum()
    ult = 100*(4*avg(bp,tr_,7)+2*avg(bp,tr_,14)+avg(bp,tr_,28))/7
    return (ult+mfi+mom+cci+rsi+willr+stosk)/7

def hesapla_stoch_rsi(df, rp=21, sp=21, k=3, d=3):
    c = df["Close"]; dlt = c.diff()
    g = dlt.clip(lower=0).rolling(rp).mean()
    ls = (-dlt.clip(upper=0)).rolling(rp).mean()
    rsi = 100-100/(1+g/ls.replace(0,np.nan))
    rm = rsi.rolling(sp).min(); rx = rsi.rolling(sp).max()
    K = ((rsi-rm)/(rx-rm+1e-10)*100).rolling(k).mean()
    return K, K.rolling(d).mean()

def hesapla_mfi(df, period=21):
    h,l,c,v = df["High"],df["Low"],df["Close"],df["Volume"]
    hlc3 = (h+l+c)/3; dif = hlc3.diff()
    up = (v*hlc3.where(dif>0,0)).rolling(period).sum()
    dn = (v*hlc3.where(dif<0,0)).rolling(period).sum()
    return 100-100/(1+up/dn.replace(0,np.nan))

def hesapla_rsi(df, period=21):
    c = df["Close"]; dlt = c.diff()
    g = dlt.clip(lower=0).rolling(period).mean()
    ls = (-dlt.clip(upper=0)).rolling(period).mean()
    return 100-100/(1+g/ls.replace(0,np.nan))

def mfi_dip_bul(mfi, v=35.0):
    """
    1 yıllık MFI verisindeki yerel dipleri bulur,
    bu diplerin kümelendiği seviyeyi KDE ile tespit eder,
    ardından o seviyenin 7 puan üstünü dip eşiği olarak döner.
    Örnek: Gerçek dipler 38 civarında kümelendiyse → eşik = 45
    """
    seri = mfi.dropna()
    if len(seri) < 30:
        return v

    # Yerel dipleri bul (her iki taraftan da düşük olan noktalar)
    values = seri.values
    yerel_dipler = []
    pencere = 5  # her iki yönde kaç bar bakılsın
    for i in range(pencere, len(values) - pencere):
        bolge = values[i - pencere: i + pencere + 1]
        if values[i] == bolge.min() and values[i] < 50:
            yerel_dipler.append(values[i])

    if len(yerel_dipler) < 3:
        # Yerel dip bulamazsak genel alt bölgeyi kullan
        alt_bolge = seri[seri < 45]
        if len(alt_bolge) < 5:
            return v
        yerel_dipler = alt_bolge.values.tolist()

    dipler = np.array(yerel_dipler)

    try:
        kde = stats.gaussian_kde(dipler)
        x = np.linspace(dipler.min(), min(dipler.max(), 50), 500)
        # En sık dip seviyesi
        en_sik_dip = float(x[np.argmax(kde(x))])
        # O seviyenin 7 puan üstü = dip eşiği
        esik = en_sik_dip + 7.0
        # 20-55 arasında sınırla
        return float(np.clip(esik, 20.0, 55.0))
    except:
        return v

# ══════════════════════════════════════════════════════
# PUANLAMA — sadece TAM PUAN ya da 0
# ══════════════════════════════════════════════════════
def puan_hesapla(tke_v, sk_v, sd_v, mfi_v, rsi_v, mfi_dip=35.0):
    def nan_kontrol(val):
        return val is None or (isinstance(val, float) and np.isnan(val))

    # TKE → dip seviyesi 20 altındaysa 30, değilse 0
    p_tke = 30.0 if (not nan_kontrol(tke_v) and tke_v <= 20) else 0.0

    # StochRSI → K ve D ortalaması 20 altındaysa 30, değilse 0
    if nan_kontrol(sk_v) or nan_kontrol(sd_v):
        p_stoch = 0.0
    else:
        stoch_ort = (sk_v + sd_v) / 2
        p_stoch = 30.0 if stoch_ort <= 20 else 0.0

    # MFI → hisse özelinde dip seviyesi altındaysa 30, değilse 0
    p_mfi = 30.0 if (not nan_kontrol(mfi_v) and mfi_v <= mfi_dip) else 0.0

    # RSI → 40 altındaysa 10, değilse 0
    p_rsi = 10.0 if (not nan_kontrol(rsi_v) and rsi_v <= 40) else 0.0

    return {
        "TKE": p_tke,
        "StochRSI": p_stoch,
        "MFI": p_mfi,
        "RSI": p_rsi,
        "TOPLAM": p_tke + p_stoch + p_mfi + p_rsi
    }

# ══════════════════════════════════════════════════════
# ANALİZ
# ══════════════════════════════════════════════════════
def analiz(hisse, df):
    try:
        tke = hesapla_tke(df)
        sk, sd = hesapla_stoch_rsi(df)
        mfi = hesapla_mfi(df)
        rsi = hesapla_rsi(df)
        md = mfi_dip_bul(mfi)
        tv = float(tke.iloc[-1]); skv = float(sk.iloc[-1]); sdv = float(sd.iloc[-1])
        mv = float(mfi.iloc[-1]); rv = float(rsi.iloc[-1]); fv = float(df["Close"].iloc[-1])
        p = puan_hesapla(tv, skv, sdv, mv, rv, md)
        return {
            "Hisse":hisse,"Fiyat":round(fv,2),
            "TKE":round(tv,2),"StochRSI_K":round(skv,2),"StochRSI_D":round(sdv,2),
            "MFI":round(mv,2),"MFI_Dip":round(md,2),"RSI":round(rv,2),
            "Puan_TKE":p["TKE"],"Puan_StochRSI":p["StochRSI"],
            "Puan_MFI":p["MFI"],"Puan_RSI":p["RSI"],"Puan_Toplam":p["TOPLAM"],
            "_tke":tke,"_sk":sk,"_sd":sd,"_mfi":mfi,"_rsi":rsi,"_close":df["Close"],
        }
    except:
        return None

# ══════════════════════════════════════════════════════
# GRAFİKLER - PLOTLY
# ══════════════════════════════════════════════════════
def pl_detay(s):
    fig = make_subplots(rows=5,cols=1,shared_xaxes=True,
        row_heights=[.30,.18,.18,.18,.16],vertical_spacing=0.03,
        subplot_titles=[s["Hisse"]+" – Fiyat (TL)","TKE (30p)","Stokastik RSI (30p)","MFI (30p)","RSI (10p)"])
    def ln(sr,nm,cl,rw,ds="solid"):
        fig.add_trace(go.Scatter(x=sr.index,y=sr.values,name=nm,
            line=dict(color=cl,width=2,dash=ds)),row=rw,col=1)
    def hl(rw,y,cl,lb):
        fig.add_hline(y=y,row=rw,col=1,line=dict(color=cl,width=1,dash="dash"),
            annotation_text=lb,annotation_font=dict(size=9,color=cl),annotation_position="right")
    ln(s["_close"],"Fiyat",R["fiyat"],1)
    ln(s["_tke"],"TKE",R["TKE"],2); hl(2,20,R["g"],"Dip(20) ✅"); hl(2,80,R["r"],"Tepe(80)")
    ln(s["_sk"],"%K",R["StochRSI"],3); ln(s["_sd"],"%D","#FF9F1C",3,ds="dot")
    hl(3,20,R["g"],"Dip(20) ✅")
    ln(s["_mfi"],"MFI",R["MFI"],4)
    hl(4,s["MFI_Dip"],R["g"],"Dip("+str(round(s["MFI_Dip"],1))+") ✅")
    hl(4,80,R["r"],"Tepe(80)")
    ln(s["_rsi"],"RSI",R["RSI"],5); hl(5,40,R["g"],"Dip(40) ✅"); hl(5,60,R["r"],"Tepe(60)")
    fig.update_layout(template="plotly_dark",height=750,paper_bgcolor=R["bg"],
        plot_bgcolor=R["panel"],font=dict(color=R["txt"],family="monospace"),
        legend=dict(orientation="h",y=-0.04),margin=dict(l=50,r=90,t=50,b=40),
        title=dict(text="📊 "+s["Hisse"]+" – İndikatör Analizi",
            font=dict(size=17,color=R["fiyat"])))
    return fig

def pl_radar(s):
    cats = ["TKE (30)","StochRSI (30)","MFI (30)","RSI (10)"]
    vals = [s["Puan_TKE"],s["Puan_StochRSI"],s["Puan_MFI"],s["Puan_RSI"]]
    maxs = [30,30,30,10]
    norm = [v/m*100 for v,m in zip(vals,maxs)]; norm.append(norm[0]); cats.append(cats[0])
    fig = go.Figure(go.Scatterpolar(r=norm,theta=cats,fill="toself",
        fillcolor="rgba(78,205,196,0.25)",line=dict(color=R["StochRSI"],width=2)))
    fig.update_layout(polar=dict(bgcolor=R["panel"],
        radialaxis=dict(visible=True,range=[0,100],color=R["txt"],tickfont=dict(size=9)),
        angularaxis=dict(color=R["txt"])),paper_bgcolor=R["bg"],
        font=dict(color=R["txt"]),showlegend=False,height=340,
        margin=dict(l=40,r=40,t=50,b=30),
        title=dict(text="🎯 "+s["Hisse"]+" – Puan Radar",font=dict(size=14,color=R["fiyat"])))
    return fig

def pl_bar(df, top_n=20):
    top = df.head(top_n).sort_values("Puan_Toplam",ascending=True)
    fig = go.Figure()
    for k,cl,lb in [("Puan_TKE",R["TKE"],"TKE (30p)"),
                    ("Puan_StochRSI",R["StochRSI"],"StochRSI (30p)"),
                    ("Puan_MFI",R["MFI"],"MFI (30p)"),
                    ("Puan_RSI",R["RSI"],"RSI (10p)")]:
        fig.add_trace(go.Bar(y=top["Hisse"],x=top[k],name=lb,
            orientation="h",marker=dict(color=cl,opacity=0.85)))
    fig.update_layout(barmode="stack",template="plotly_dark",paper_bgcolor=R["bg"],
        plot_bgcolor=R["panel"],font=dict(color=R["txt"],family="monospace"),
        xaxis=dict(title="Toplam Puan (maks 100)",range=[0,100]),
        title=dict(text="🏆 En Yüksek Puanlı "+str(top_n)+" Hisse",
            font=dict(size=16,color=R["fiyat"])),
        legend=dict(orientation="h",y=-0.12),height=max(400,top_n*28),
        margin=dict(l=70,r=30,t=60,b=80))
    return fig

def pl_heat(df, top_n=30):
    top = df.head(top_n)
    kols = ["Puan_TKE","Puan_StochRSI","Puan_MFI","Puan_RSI","Puan_Toplam"]
    etk = ["TKE","StochRSI","MFI","RSI","TOPLAM"]
    z = top[kols].values.tolist()
    fig = go.Figure(go.Heatmap(z=z,x=etk,y=top["Hisse"].tolist(),
        colorscale="RdYlGn",zmin=0,zmax=100,
        text=[[str(round(v,0)) for v in row] for row in z],
        texttemplate="%{text}",textfont=dict(size=10),colorbar=dict(title="Puan")))
    fig.update_layout(template="plotly_dark",paper_bgcolor=R["bg"],plot_bgcolor=R["panel"],
        font=dict(color=R["txt"]),height=max(500,top_n*22),margin=dict(l=80,r=30,t=60,b=40),
        title=dict(text="🌡️ Isı Haritası – İlk "+str(top_n)+" Hisse",
            font=dict(size=16,color=R["fiyat"])))
    return fig

# ══════════════════════════════════════════════════════
# GRAFİKLER - MATLOTLİB
# ══════════════════════════════════════════════════════
def mpl_detay(s):
    fig = plt.figure(figsize=(14,10),facecolor=R["bg"])
    gs = gridspec.GridSpec(5,1,figure=fig,hspace=0.08,height_ratios=[2.5,1.2,1.2,1.2,1.0])
    axs = [fig.add_subplot(gs[i]) for i in range(5)]
    for ax in axs:
        ax.set_facecolor(R["panel"]); ax.tick_params(colors=R["txt"],labelsize=8)
        ax.spines[:].set_color("#333355")
    axs[0].plot(range(len(s["_close"])),s["_close"].values,color=R["fiyat"],lw=1.5)
    axs[0].set_ylabel("Fiyat",color=R["txt"],fontsize=9)
    axs[0].set_title(s["Hisse"]+" – Toplam: "+str(s["Puan_Toplam"])+"/100",
        color=R["fiyat"],fontsize=12)
    for ax,sr,cl,lb,dp,tp in [
        (axs[1],s["_tke"],R["TKE"],"TKE",20,80),
        (axs[3],s["_mfi"],R["MFI"],"MFI",s["MFI_Dip"],80),
        (axs[4],s["_rsi"],R["RSI"],"RSI",40,60)]:
        ax.plot(range(len(sr)),sr.values,color=cl,lw=1.5)
        ax.axhline(dp,color=R["g"],lw=0.8,ls="--"); ax.axhline(tp,color=R["r"],lw=0.8,ls="--")
        ax.set_ylabel(lb,color=R["txt"],fontsize=9); ax.set_ylim(0,100)
    axs[2].plot(range(len(s["_sk"])),s["_sk"].values,color=R["StochRSI"],lw=1.5,label="%K")
    axs[2].plot(range(len(s["_sd"])),s["_sd"].values,color="#FF9F1C",lw=1,ls=":",label="%D")
    axs[2].axhline(20,color=R["g"],lw=0.8,ls="--")
    axs[2].set_ylabel("StochRSI",color=R["txt"],fontsize=9); axs[2].set_ylim(0,100)
    axs[2].legend(fontsize=7,facecolor=R["panel"],labelcolor=R["txt"],loc="upper left")
    for ax in axs[:-1]: ax.tick_params(labelbottom=False)
    axs[4].set_xlabel("Gün",color=R["txt"],fontsize=8)
    plt.tight_layout(); return fig

def mpl_puan(s):
    fig, ax = plt.subplots(figsize=(6,3),facecolor=R["bg"]); ax.set_facecolor(R["panel"])
    cats = ["TKE\n(30p)","StochRSI\n(30p)","MFI\n(30p)","RSI\n(10p)"]
    ps = [s["Puan_TKE"],s["Puan_StochRSI"],s["Puan_MFI"],s["Puan_RSI"]]
    cls = [R["TKE"],R["StochRSI"],R["MFI"],R["RSI"]]
    bars = ax.bar(cats,ps,color=cls,edgecolor="none",width=0.55)
    for b,p in zip(bars,ps):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,
            str(int(p)),ha="center",va="bottom",color=R["txt"],fontsize=9)
    ax.set_ylim(0,35); ax.tick_params(colors=R["txt"]); ax.spines[:].set_color("#333355")
    ax.set_title(s["Hisse"]+" – Toplam: "+str(int(s["Puan_Toplam"]))+"/100",
        color=R["fiyat"],fontsize=10)
    fig.tight_layout(); return fig

# ══════════════════════════════════════════════════════
# STREAMLIT ARAYÜZ
# ══════════════════════════════════════════════════════
st.set_page_config(page_title="BIST 100 Dip Bulucu",page_icon="📉",
    layout="wide",initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
html,body,[class*="css"]{font-family:'Syne',sans-serif;background:#0f0f1a;color:#e0e0e0;}
.stApp{background-color:#0f0f1a;}
h1{font-size:2rem!important;color:#FF6B35!important;}
h2{color:#4ECDC4!important;} h3{color:#FFE66D!important;}
.kart{background:#16213E;border:1px solid #1a1a3e;border-radius:12px;padding:16px 20px;text-align:center;}
.kart-val{font-size:1.9rem;font-weight:700;font-family:'JetBrains Mono',monospace;}
.kart-lbl{font-size:0.72rem;color:#888;margin-top:4px;}
.yesil{color:#06D6A0;}.sari{color:#FFE66D;}.kirmizi{color:#EF476F;}
.stButton>button{background:linear-gradient(135deg,#FF6B35,#FF9F1C);color:white;
    font-weight:700;border:none;border-radius:8px;padding:.5rem 1.5rem;}
</style>""",unsafe_allow_html=True)

st.markdown("# 📉 BIST 100 Dip Bulucu")
st.markdown("**TKE · Stokastik RSI · MFI · RSI** — Dip seviyesinde olan hisseleri puanlar")
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    periyot = st.selectbox("Veri Periyodu",["6mo","1y","2y"],index=1,
        format_func=lambda x:{"6mo":"6 Ay","1y":"1 Yıl","2y":"2 Yıl"}[x])
    min_puan = st.slider("Minimum Puan Filtresi",0,100,40,step=10)
    top_n = st.slider("Gösterilecek Hisse Sayısı",5,50,20,step=5)
    st.divider()
    st.markdown("**Puanlama Sistemi**")
    st.caption("Her indikatör ya tam puan ya 0 verir:")
    st.caption("TKE < 20 → **30p** | yoksa 0")
    st.caption("StochRSI K+D ort < 20 → **30p** | yoksa 0")
    st.caption("MFI < dip seviyesi → **30p** | yoksa 0")
    st.caption("RSI < 40 → **10p** | yoksa 0")
    st.caption("Maksimum: **100p**")
    st.divider()
    tara_btn = st.button("🔍 Tüm BIST 100'ü Tara",use_container_width=True)
    st.divider()
    st.markdown("**Tek Hisse Analizi**")
    tek_hisse = st.selectbox("Hisse Seç",sorted(BIST100))
    tek_btn = st.button("📈 Analiz Et",use_container_width=True)

for k in ["df_sonuc","sonuclar","tek_sonuc"]:
    if k not in st.session_state: st.session_state[k] = None

def goster(s):
    st.markdown("## 📈 "+s["Hisse"]+" – Detay Analizi")
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(lb,val,mx) in zip([c1,c2,c3,c4,c5],[
        ("TOPLAM",s["Puan_Toplam"],100),("TKE",s["Puan_TKE"],30),
        ("StochRSI",s["Puan_StochRSI"],30),("MFI",s["Puan_MFI"],30),("RSI",s["Puan_RSI"],10)]):
        r = "yesil" if val==mx else "kirmizi"
        ikon = " ✅" if val==mx else " ❌"
        col.markdown('<div class="kart"><div class="kart-val '+r+'">'+str(int(val))+
            '</div><div class="kart-lbl">'+lb+ikon+' / '+str(mx)+'</div></div>',
            unsafe_allow_html=True)
    st.markdown("")
    ca,cb = st.columns([2,1])
    with ca: st.plotly_chart(pl_detay(s),use_container_width=True)
    with cb:
        st.plotly_chart(pl_radar(s),use_container_width=True)
        st.pyplot(mpl_puan(s))
    with st.expander("🖨️ PNG İndir"):
        fig2 = mpl_detay(s); buf = io.BytesIO()
        fig2.savefig(buf,format="png",dpi=150,bbox_inches="tight"); buf.seek(0)
        st.image(buf)
        st.download_button("⬇️ PNG İndir",data=buf,
            file_name=s["Hisse"]+"_analiz.png",mime="image/png")

if tara_btn:
    st.info("📡 Veriler çekiliyor... (~2 dk)")
    prog = st.progress(0); txt = st.empty()
    veriler = {}; top = len(BIST100)
    for i,h in enumerate(BIST100):
        d = veri_cek(h,periyot)
        if d is not None: veriler[h] = d
        prog.progress((i+1)/top)
        txt.text("⏳ "+h+" ("+str(i+1)+"/"+str(top)+")")
        time.sleep(0.04)
    txt.text("📊 Puanlama hesaplanıyor...")
    sonuclar = [analiz(h,d) for h,d in veriler.items()]
    sonuclar = [s for s in sonuclar if s]
    df_s = pd.DataFrame([{k:v for k,v in s.items() if not k.startswith("_")} for s in sonuclar])
    df_s = df_s.sort_values("Puan_Toplam",ascending=False).reset_index(drop=True)
    prog.empty(); txt.empty()
    st.session_state.df_sonuc = df_s
    st.session_state.sonuclar = sonuclar
    st.success("✅ "+str(len(df_s))+" hisse analiz edildi!")

if st.session_state.df_sonuc is not None:
    df = st.session_state.df_sonuc
    df_f = df[df["Puan_Toplam"] >= min_puan]
    st.markdown("## 🏆 Dip Puanlama Sonuçları")
    c1,c2,c3,c4 = st.columns(4)
    t1 = df.iloc[0]
    for col,val,lb in [
        (c1,str(len(df_f)),"Filtreden Geçen"),
        (c2,str(t1["Hisse"]),"En İyi: "+str(int(t1["Puan_Toplam"]))+"p"),
        (c3,str(int(df["Puan_Toplam"].mean())),"Ortalama Puan"),
        (c4,str((df["Puan_Toplam"]>=60).sum()),"60+ Puan")]:
        col.markdown('<div class="kart"><div class="kart-val sari">'+val+
            '</div><div class="kart-lbl">'+lb+'</div></div>',unsafe_allow_html=True)
    st.markdown("")
    t1,t2,t3 = st.tabs(["📊 Bar Sıralama","🌡️ Isı Haritası","📋 Tablo"])
    with t1: st.plotly_chart(pl_bar(df_f,top_n),use_container_width=True)
    with t2: st.plotly_chart(pl_heat(df_f,top_n),use_container_width=True)
    with t3:
        kols = ["Hisse","Fiyat","Puan_Toplam","TKE","Puan_TKE",
                "StochRSI_K","StochRSI_D","Puan_StochRSI",
                "MFI","MFI_Dip","Puan_MFI","RSI","Puan_RSI"]
        st.dataframe(df_f[kols].style.background_gradient(
            subset=["Puan_Toplam"],cmap="RdYlGn").format(precision=2),
            use_container_width=True,height=500)
        csv = df_f[kols].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ CSV İndir",data=csv,
            file_name="bist100_dip.csv",mime="text/csv")
    if st.session_state.sonuclar:
        st.divider(); st.markdown("### 🔎 Listeden Hisse Detayı")
        sec = st.selectbox("Hisse",[s["Hisse"] for s in st.session_state.sonuclar],key="ls")
        ss = next((s for s in st.session_state.sonuclar if s["Hisse"]==sec),None)
        if ss: goster(ss)

if tek_btn:
    with st.spinner("📡 "+tek_hisse+" çekiliyor..."):
        df_t = veri_cek(tek_hisse,periyot)
    if df_t is None:
        st.error("❌ "+tek_hisse+" için veri alınamadı.")
    else:
        s = analiz(tek_hisse,df_t)
        if s: st.session_state.tek_sonuc = s
        else: st.error("❌ Analiz hesaplanamadı.")

if st.session_state.tek_sonuc and not tara_btn:
    goster(st.session_state.tek_sonuc)

st.divider()
st.markdown("""<div style='text-align:center;color:#444;font-size:.75rem;'>
BIST 100 Dip Bulucu · TKE (Kıvanç Özbilgiç) · Stokastik RSI · MFI · RSI<br>
⚠️ Yatırım tavsiyesi değildir. Veriler Yahoo Finance'den çekilmektedir.
</div>""",unsafe_allow_html=True)
