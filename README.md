# SPK CBR - Prediksi Risiko Diabetes (Pima Indians)

Deskripsi:
Aplikasi web sederhana yang mengimplementasikan Case-Based Reasoning (CBR) untuk memprediksi risiko diabetes menggunakan dataset Pima Indians.

File penting:
- app.py           (Streamlit app)
- cbr.py           (logika CBR: retrieve, reuse, revise, retain, evaluasi)
- utils.py         (download dataset)
- requirements.txt

Dataset:
- URL raw CSV: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

Cara jalan (lokal):
1. Clone / ekstrak folder ini.
2. Buat environment, install dependencies:
   - `pip install -r requirements.txt`
3. Jalankan:
   - `streamlit run app.py`
4. Buka browser ke `http://localhost:8501`

Deploy ke Streamlit Community Cloud:
1. Push seluruh folder ke GitHub (root berisi app.py & requirements.txt)
2. Buka https://share.streamlit.io → Connect GitHub → pilih repo & file app.py → Deploy
3. Salin URL aplikasi publik dan lampirkan pada halaman tugas.

Catatan:
- Sistem CBR ini sederhana dan cocok untuk demonstrasi akademis.
- Untuk produksi, tambahkan validasi domain, keamanan data, dan manajemen versi basis kasus.
