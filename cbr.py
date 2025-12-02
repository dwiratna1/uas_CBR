import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter

class CBRSystem:
    def __init__(self, df_cases, feature_cols, label_col='Outcome', scaler=None):
        """
        df_cases: DataFrame berisi kasus historis (harus memuat feature_cols + label_col)
        feature_cols: list nama kolom fitur numerik
        label_col: nama kolom label (0/1)
        """
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.df_cases = df_cases.copy().reset_index(drop=True)
        # scaler: jika None, buat StandardScaler dan fit pada kasus
        if scaler is None:
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(self.df_cases[self.feature_cols].values)
        else:
            self.scaler = scaler
            self.X_scaled = self.scaler.transform(self.df_cases[self.feature_cols].values)
    
    def retrieve(self, query, k=5, metric='euclidean', weight_by_distance=False):
        """
        query: dict atau DataFrame 1 row dengan feature_cols
        k: jumlah tetangga terdekat
        metric: 'euclidean' atau 'manhattan' (saat ini)
        weight_by_distance: kalau True, hasil voting diberi bobot 1/d (d=distance)
        Return: DataFrame kasus terpilih (k kasus), distances array, labels array
        """
        # prepare query vector
        if isinstance(query, dict):
            q = np.array([query[feat] for feat in self.feature_cols], dtype=float).reshape(1,-1)
        elif isinstance(query, pd.DataFrame):
            q = query[self.feature_cols].values.astype(float)
            if q.shape[0] != 1:
                raise ValueError("Query DataFrame harus 1 baris")
        else:
            q = np.array(query).reshape(1,-1).astype(float)
        
        q_scaled = self.scaler.transform(q)  # shape (1, n_features)
        # compute distances
        if metric == 'euclidean':
            dists = np.linalg.norm(self.X_scaled - q_scaled, axis=1)
        elif metric == 'manhattan':
            dists = np.sum(np.abs(self.X_scaled - q_scaled), axis=1)
        else:
            raise ValueError("Metric tidak dikenal")
        # pick k smallest
        idx = np.argsort(dists)[:k]
        neighbors = self.df_cases.iloc[idx].copy().reset_index(drop=True)
        neighbors['_distance'] = dists[idx]
        return neighbors, dists[idx]
    
    def reuse(self, neighbors, dists=None, weight_by_distance=False):
        """
        neighbors: DataFrame hasil retrieve (harus berisi label_col)
        dists: distances corresponding to neighbors (optional)
        weight_by_distance: kalau True pakai bobot 1/(d+epsilon)
        Return: predicted label (0/1), detail voting
        """
        labels = neighbors[self.label_col].astype(int).values
        if weight_by_distance and dists is not None:
            eps = 1e-6
            weights = 1.0 / (dists + eps)
            # aggregate weighted vote per class
            classes = np.unique(labels)
            vote = {}
            for c in classes:
                vote[c] = weights[labels == c].sum()
            # pilih class dengan bobot terbesar
            pred = max(vote.items(), key=lambda x: x[1])[0]
            return int(pred), vote
        else:
            cnt = Counter(labels)
            pred = cnt.most_common(1)[0][0]
            # also provide counts
            return int(pred), dict(cnt)
    
    def revise_and_retain(self, query_row, proven_label, save_path=None):
        """
        Tambah kasus baru ke basis kasus (retain). proven_label = ground truth (0/1)
        query_row: dict atau 1-row DataFrame berisi fitur
        save_path: jika tidak None, simpan ke CSV (append)
        """
        if isinstance(query_row, dict):
            row = {**query_row}
        elif isinstance(query_row, pd.Series):
            row = query_row.to_dict()
        elif isinstance(query_row, pd.DataFrame):
            row = query_row.iloc[0].to_dict()
        else:
            raise ValueError("query_row harus dict / Series / DataFrame(1 row)")
        row[self.label_col] = int(proven_label)
        # append to df_cases
        self.df_cases = pd.concat([self.df_cases, pd.DataFrame([row])], ignore_index=True)
        # Update scaled matrix
        self.X_scaled = self.scaler.fit_transform(self.df_cases[self.feature_cols].values)
        if save_path is not None:
            # Simpan seluruh basis kasus (overwrite)
            self.df_cases.to_csv(save_path, index=False)
        return True

    def evaluate_leave_one_out(self, k=5, metric='euclidean'):
        """
        Evaluasi sederhana: leave-one-out CBR: untuk setiap kasus i, gunakan semua kasus lain sebagai basis,
        retrieve k neighbors, lakukan majority vote -> bandingkan pred vs actual.
        Return: akurasi, detail per-case (DataFrame)
        """
        n = len(self.df_cases)
        preds = []
        for i in range(n):
            # basis tanpa i
            basis_df = self.df_cases.drop(index=i).reset_index(drop=True)
            cbr_temp = CBRSystem(basis_df, self.feature_cols, self.label_col, scaler=None)
            # query = row i features
            q_dict = {feat: float(self.df_cases.loc[i, feat]) for feat in self.feature_cols}
            neighbors, dists = cbr_temp.retrieve(q_dict, k=k, metric=metric)
            pred, _ = cbr_temp.reuse(neighbors, dists=dists, weight_by_distance=False)
            preds.append(pred)
        actuals = self.df_cases[self.label_col].astype(int).values
        preds = np.array(preds)
        accuracy = (preds == actuals).mean()
        return accuracy
