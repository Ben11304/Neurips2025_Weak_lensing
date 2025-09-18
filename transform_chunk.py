def compute_global_image_stats(chunk_dir: str, indices: list) -> Tuple[float, float]:
    """
    Lặp qua toàn bộ noisy kappa từ tất cả chunk để tính mean và std toàn cục.
    Sử dụng online computation để tránh load hết dữ liệu.
    """
    total_sum = 0.0
    total_sum_sq = 0.0  # Để tính variance
    total_count = 0
    
    for chunk_idx in indices:
        noisy_chunk, _, _ = next(iter_chunks(chunk_dir, indices=[chunk_idx]))
        # Reshape để tính trên tất cả pixel (flatten theo batch và spatial dims)
        flat_data = noisy_chunk.reshape(-1, noisy_chunk.shape[-2] * noisy_chunk.shape[-1])  # (N_total_pixels_per_chunk,)
        n_pixels = flat_data.size
        total_sum += np.sum(flat_data, dtype=np.float64)
        total_sum_sq += np.sum(flat_data**2, dtype=np.float64)
        total_count += n_pixels
    
    global_mean = total_sum / total_count
    global_var = (total_sum_sq / total_count) - (global_mean ** 2)
    global_std = np.sqrt(global_var)
    
    return global_mean.astype(np.float32), global_std.astype(np.float32)

def compute_global_label_stats(chunk_dir: str, indices: list) -> StandardScaler:
    """
    Lặp qua toàn bộ labels từ tất cả chunk để fit StandardScaler toàn cục.
    """
    all_labels = []
    for chunk_idx in indices:
        _, label_chunk, _ = next(iter_chunks(chunk_dir, indices=[chunk_idx]))
        # Chỉ lấy 2 params đầu (như mã của bạn), reshape để ghép
        labels_subset = label_chunk[:, :, :2].reshape(-1, 2)  # (N_total_samples, 2)
        all_labels.append(labels_subset)
    
    all_labels = np.vstack(all_labels)  # Ghép tất cả (có thể tốn RAM nếu labels lớn, nhưng labels thường nhỏ)
    
    scaler = StandardScaler()
    scaler.fit(all_labels)
    return scaler