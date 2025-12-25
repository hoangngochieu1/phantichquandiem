import streamlit as st
import torch
import json
import os
import gdown
import numpy as np
from transformers import AutoTokenizer
from model import JointACDSPCModel
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACT_DIR = "/absa_prepared"

# Tải thẳng file .pt vào root app
MODEL_PATH = "joint_acd_spc_model_final.pt"
GDRIVE_FILE_URL = "https://drive.google.com/uc?id=1p99F1BKmL6mEZdPcDFzfQjN4Pv37UF51"  # link trực tiếp

print("Model exists?", os.path.exists(MODEL_PATH))

def download_model_from_drive():
    """Tải model từ Google Drive nếu chưa có"""
    if not os.path.exists(MODEL_PATH):
        gdown.download(
            url=GDRIVE_FILE_URL,
            output=MODEL_PATH,
            quiet=False,
            fuzzy=True
        )

# ---------- Load artifacts ----------
@st.cache_resource
def load_all():
    # 1️⃣ tải model .pt (nếu chưa có)
    download_model_from_drive()

    # 2️⃣ load meta
    with open(f"{ARTIFACT_DIR}/meta.json") as f:
        meta = json.load(f)

    with open(f"{ARTIFACT_DIR}/model_kwargs.json") as f:
        model_kwargs = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(f"{ARTIFACT_DIR}/tokenizer")

    model = JointACDSPCModel(**model_kwargs)
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return model, tokenizer, meta


model, tokenizer, meta = load_all()



categories = meta["categories"]
idx2cat = {int(k): v for k, v in meta["idx2cat"].items()}
MAX_LEN = meta["max_len"]
THRESHOLD = meta["threshold"]
df = None


sentiment_map = {0: "neutral", 1: "positive", 2: "negative"}

# ---------- Prediction ----------
def predict(text, threshold):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        acd_logits, spc_logits = model(**enc)

    acd_probs = torch.sigmoid(acd_logits)[0].cpu().numpy()
    spc_logits = spc_logits[0].view(len(categories), 3)
    spc_probs = torch.softmax(spc_logits, dim=1).cpu().numpy()
    spc_preds = np.argmax(spc_probs, axis=1)

    results = []
    for i, p in enumerate(acd_probs):
        if p >= threshold:
            results.append({
                "aspect": idx2cat[i],
                "sentiment": sentiment_map[spc_preds[i]],
                "confidence": float(p),
                "sentiment_probs": spc_probs[i]
            })
    return results

# Sidebar – control panel
st.set_page_config(page_title="ABSA Demo", layout="centered")
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 95%;
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.sidebar.title("⚙️ Settings")

threshold = st.sidebar.slider(
    "ACD Threshold",
    min_value=0.1,
    max_value=0.9,
    value=THRESHOLD,
    step=0.05
)

show_details = st.sidebar.checkbox("Show detailed scores", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:**")
st.sidebar.write(meta["model_name"])
st.sidebar.write(f"Num aspects: {len(categories)}")




# ---------- Streamlit UI ----------
st.set_page_config(page_title="ABSA Demo", layout="centered")

st.title("🔍 Aspect-Based Sentiment Analysis")
st.write("Nhập phản hồi khách hàng để phân tích **aspect & sentiment**")

text = st.text_area("✍️ Nhập câu:", height=120)

if st.button("🚀 Phân tích"):
    if not text.strip():
        st.warning("Vui lòng nhập nội dung.")
    else:
        with st.spinner("Đang phân tích..."):
            results = predict(text, threshold)

        if len(results) == 0:
            st.info("⚠️ Không phát hiện aspect nào vượt ngưỡng.")
        else:
            st.success("✅ Kết quả phân tích")

            for r in results:
                color = {
                    "positive": "green",
                    "negative": "red",
                    "neutral": "orange"
                }[r["sentiment"]]

                st.markdown(
                    f"""
                    **{r['aspect']}**  
                    <span style="color:{color}; font-weight:bold">
                    {r['sentiment']}
                    </span>  
                    Confidence: `{r['confidence']:.2f}`
                    """,
                    unsafe_allow_html=True
                )

                if show_details:
                    st.write(
                        f"• Sentiment probs → "
                        f"Pos: {r['sentiment_probs'][1]:.2f}, "
                        f"Neu: {r['sentiment_probs'][0]:.2f}, "
                        f"Neg: {r['sentiment_probs'][2]:.2f}"
                    )

                st.markdown("---")

def batch_predict(sentences, threshold):
    rows = []

    for sid, text in enumerate(sentences, 1):
        text = text.strip()
        if not text:
            continue

        preds = predict(text, threshold)

        if len(preds) == 0:
            rows.append({
                "sentence_id": sid,
                "sentence": text,
                "aspect": None,
                "sentiment": None,
                "confidence": None
            })
        else:
            for r in preds:
                rows.append({
                    "sentence_id": sid,
                    "sentence": text,
                    "aspect": r["aspect"],
                    "sentiment": r["sentiment"],
                    "confidence": r["confidence"]
                })

    return rows

# 1 option 2 Batch inference cho nhiều câu
def batch_predict(sentences, threshold):
    rows = []

    for sid, text in enumerate(sentences, 1):
        text = text.strip()
        if not text:
            continue

        preds = predict(text, threshold)

        if len(preds) == 0:
            rows.append({
                "sentence_id": sid,
                "sentence": text,
                "aspect": None,
                "sentiment": None,
                "confidence": None
            })
        else:
            for r in preds:
                rows.append({
                    "sentence_id": sid,
                    "sentence": text,
                    "aspect": r["aspect"],
                    "sentiment": r["sentiment"],
                    "confidence": r["confidence"]
                })

    return rows

# 2 Upload .txt
st.subheader("📂 Upload file .txt (mỗi dòng = 1 câu)")
uploaded_file = st.file_uploader("Chọn file .txt", type=["txt"])


# # 3 Xử lý khi upload

if uploaded_file is not None:
    raw_text = uploaded_file.read().decode("utf-8")
    sentences = raw_text.splitlines()

    with st.spinner("🔍 Đang phân tích..."):
        results = batch_predict(sentences, threshold)

    df = pd.DataFrame(results)


    # 4 Hiển thị bảng kết quả
    st.subheader("📋 Bảng kết quả")

    st.dataframe(
        df.fillna("—"),
        use_container_width=True
    )
    # # Thống kê aspect & sentiment

        # ===============================
    # BUILD pivot_df 
    # ===============================

    aspect_sentiment_df = (
        df[df["aspect"].notna() & df["sentiment"].notna()]
        .groupby(["aspect", "sentiment"])
        .size()
        .reset_index(name="count")
    )

    pivot_df = aspect_sentiment_df.pivot(
        index="aspect",
        columns="sentiment",
        values="count"
    ).fillna(0)



    # Chuyển pivot về dạng long
    plot_df = (
        pivot_df
        .reset_index()
        .melt(id_vars="aspect", var_name="sentiment", value_name="count")
    )

    # Tạo thứ tự aspect theo tổng count
    aspect_order = (
        pivot_df.sum(axis=1)
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "aspect:N",
                sort=aspect_order,        # 🔥 ÉP THỨ TỰ TẠI ĐÂY
                title="Aspect"
            ),
            y=alt.Y(
                "count:Q",
                title="Count"
            ),
            color=alt.Color(
                "sentiment:N",
                scale=alt.Scale(
                    domain=["positive", "neutral", "negative"],
                    range=["#2ecc71", "#f1c40f", "#e74c3c"]
                ),
                title="Sentiment"
            ),
            tooltip=["aspect", "sentiment", "count"]
        )
        .properties(height=400)
    )

    st.altair_chart(chart, use_container_width=True)


    
    # # Thống kê aspect & sentiment
    # aspect_sentiment_df = (
    #     df[df["aspect"].notna() & df["sentiment"].notna()]
    #     .groupby(["aspect", "sentiment"])
    #     .size()
    #     .reset_index(name="count")
    # )

    # pivot_df = aspect_sentiment_df.pivot(
    #     index="aspect",
    #     columns="sentiment",
    #     values="count"
    # ).fillna(0)

    # TOP_K = 10

    # # Lấy TOP_K aspect nhiều nhất
    # pivot_df = pivot_df.loc[
    #     pivot_df.sum(axis=1).sort_values(ascending=False).head(TOP_K).index
    # ]

    # # 🔥 SẮP XẾP LẠI THEO THỨ TỰ GIẢM DẦN
    # pivot_df = pivot_df.loc[
    #     pivot_df.sum(axis=1).sort_values(ascending=False).index
    # ]

    # st.subheader("📊 Aspect × Sentiment distribution")
    # st.bar_chart(pivot_df)

        

    # 5 SUMMARY
    # Thống kê % câu có aspect
    total_sent = df["sentence_id"].nunique()
    sent_with_aspect = df[df["aspect"].notna()]["sentence_id"].nunique()

    st.metric(
        label="📌 % câu có aspect",
        value=f"{sent_with_aspect / total_sent * 100:.2f}%"
    )




    # 📊 Sentiment distribution
    st.subheader("📊 Sentiment distribution")

    sentiment_counts = (
        df[df["sentiment"].notna()]
        ["sentiment"]
        .value_counts()
    )

    # sắp xếp theo thứ tự mong muốn
    sentiment_order = ["positive", "neutral", "negative"]

    sentiment_counts = sentiment_counts.reindex(sentiment_order).fillna(0)

    st.bar_chart(sentiment_counts)



    # 6 DOWNLOAD CSV & JSON
    st.subheader("⬇️ Download kết quả")

    csv_data = df.to_csv(index=False).encode("utf-8")
    json_data = df.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "⬇ Download CSV",
            csv_data,
            file_name="absa_results.csv",
            mime="text/csv"
        )

    with col2:
        st.download_button(
            "⬇ Download JSON",
            json_data,
            file_name="absa_results.json",
            mime="application/json"
        )



## khuyến nghị
def compute_aspect_stats(df):
    stats = (
        df[df["aspect"].notna() & df["sentiment"].notna()]
        .groupby(["aspect", "sentiment"])
        .size()
        .unstack(fill_value=0)
    )

    # đảm bảo đủ 3 cột
    for col in ["positive", "negative", "neutral"]:
        if col not in stats.columns:
            stats[col] = 0

    stats["total"] = stats.sum(axis=1)

    stats["positive_ratio"] = stats["positive"] / stats["total"]
    stats["negative_ratio"] = stats["negative"] / stats["total"]
    stats["neutral_ratio"]  = stats["neutral"]  / stats["total"]

    return stats.reset_index()

    
def generate_recommendations(aspect_stats):
    recommendations = []

    for _, row in aspect_stats.iterrows():
        aspect = row["aspect"]
        total = row["total"]

        if total < 5:
            continue  # bỏ qua aspect quá ít dữ liệu

        pos = row["positive_ratio"]
        neg = row["negative_ratio"]
        neu = row["neutral_ratio"]

        if neg >= 0.4:
            rec = f"⚠️ **{aspect}** có tỷ lệ phản hồi tiêu cực cao ({neg:.0%}). Nên ưu tiên cải thiện."
        elif pos >= 0.6:
            rec = f"⭐ **{aspect}** là điểm mạnh ({pos:.0%} phản hồi tích cực). Nên duy trì và quảng bá."
        elif neu >= 0.5:
            rec = f"🤔 **{aspect}** có nhiều phản hồi trung tính. Có thể khách hàng chưa cảm nhận rõ, cần cải thiện trải nghiệm."
        else:
            rec = f"ℹ️ **{aspect}** có phản hồi tương đối cân bằng."

        recommendations.append(rec)

    return recommendations

if df is not None and len(df) > 0:

    aspect_stats = compute_aspect_stats(df)

    #bảng phân tích

    st.subheader("📋 Aspect sentiment summary")

    sorted_df = aspect_stats.sort_values("total", ascending=False)

    st.dataframe(
        sorted_df[
            [
                "aspect",
                "positive",
                "negative",
                "neutral",
                "total",
                "positive_ratio",
                "negative_ratio",
                "neutral_ratio"
            ]
        ],
        height=500
    )

    # 7 Khuyến nghị kinh doanh
        
    st.subheader("🧠 Business Recommendations")


    recommendations = generate_recommendations(aspect_stats)

    if len(recommendations) == 0:
        st.info("Không đủ dữ liệu để đưa ra khuyến nghị.")
    else:
        for r in recommendations:
            st.write(r)


