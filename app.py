import streamlit as st
import pandas as pd
import numpy as np
from utils import ensure_data, COL_NAMES
from cbr import CBRSystem

st.set_page_config(page_title="SPK CBR - Prediksi Diabetes", layout="centered")
st.title("Sistem Pendukung Keputusan (CBR) — Prediksi Risiko Diabetes")
st.markdown("Aplikasi ini menggunakan metode **Case-Based Reasoning (CBR)** pada dataset Pima Indians untuk membantu memperkirakan apakah seseorang berisiko diabetes berdasarkan fitur medis.")

# Load data
df = ensure_data()
st.sidebar.header("Pengaturan")
st.sidebar.write("Dataset Pima loaded. Jumlah kasus:", df.shape[0])

# Pilih fitur (tampilkan default semua fitur kecuali Outcome)
feature_cols = COL_NAMES[:-1]  # semua fitur kecuali Outcome
selected_features = st.sidebar.multiselect("Pilih fitur yang akan digunakan (CBR)", feature_cols, default=feature_cols)

k_default = st.sidebar.slider("Nilai k (jumlah kasus mirip yang diambil)", min_value=1, max_value=20, value=5)
metric = st.sidebar.selectbox("Metric distance", options=['euclidean','manhattan'], index=0)
weight_opt = st.sidebar.checkbox("Berikan bobot berdasarkan jarak (1/d)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("Fitur evaluasi")
if st.sidebar.button("Evaluasi (Leave-one-out)"):
    with st.spinner("Melakukan evaluasi LOOCV... (bisa memakan waktu)"):
        cbr_all = CBRSystem(df, feature_cols=selected_features)
        acc = cbr_all.evaluate_leave_one_out(k=k_default, metric=metric)
    st.success(f"Akurasi LOOCV (k={k_default}): {acc:.4f}")

st.write("### Preview data (5 baris pertama)")
st.dataframe(df.head())

st.write("## Form Input Kasus Baru")
with st.form("form_query"):
    st.write("Isi nilai fitur pasien:")
    inputs = {}
    for feat in selected_features:
        # automatic numeric input
        val = st.number_input(feat, value=float(df[feat].median()))
        inputs[feat] = val
    submitted = st.form_submit_button("Cari kasus mirip / Prediksi")
    
if submitted:
    # Run CBR
    cbr = CBRSystem(df, feature_cols=selected_features)
    neighbors, dists = cbr.retrieve(inputs, k=k_default, metric=metric)
    pred, vote_detail = cbr.reuse(neighbors, dists=dists, weight_by_distance=weight_opt)
    
    st.write("### Hasil Prediksi (CBR)")
    st.info(f"Prediksi Outcome (0 = Tidak diabetes, 1 = Diabetes): **{pred}**")
    st.write("Voting detail:", vote_detail)
    
    st.write("### Kasus terpilih (nearest neighbors)")
    st.dataframe(neighbors.style.format({'_distance': '{:.4f}'}))
    
    # opsi revise & retain
    st.write("### Revise & Retain (opsional)")
    st.write("Jika Anda mengetahui label asli pasien (mis. hasil pemeriksaan), Anda dapat menyimpan kasus baru ke basis kasus.")
    col1, col2 = st.columns(2)
    with col1:
        gt = st.selectbox("Ground truth (jika diketahui)", options=[-1,0,1], format_func=lambda x: "Belum diketahui" if x==-1 else str(x))
    with col2:
        save_path = "data/pima-indians-diabetes.csv"
        if st.button("Simpan kasus baru ke basis kasus (retain)"):
            if gt == -1:
                st.error("Masukkan label ground truth (0 atau 1) untuk menyimpan.")
            else:
                cbr.revise_and_retain(inputs, proven_label=gt, save_path=save_path)
                st.success("Kasus berhasil disimpan (file data diperbarui).")
                # reload df for preview
                df = pd.read_csv(save_path)
                st.experimental_rerun()

st.write("---")
st.markdown("**Catatan:** Metode CBR di sini sederhana: mencari k kasus terdekat (berdasarkan fitur yang dipilih), lalu majority vote sebagai solusi. Anda bisa memperkaya sistem ini dengan weighting fitur, normalisasi yang berbeda, atau pengambilan kasus berbasis similarity thresholds.")
