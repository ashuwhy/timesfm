import time
import torch
import numpy as np
import timesfm

def generate_synthetic_data(batch_size, context_len, horizon_len):
    """Generates synthetic sine waves with noise."""
    t = np.linspace(0, 4 * np.pi, context_len + horizon_len)
    data = []
    ground_truth = []
    
    for _ in range(batch_size):
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)
        noise = np.random.normal(0, 0.1, size=t.shape)
        
        signal = np.sin(freq * t + phase)
        noisy_signal = signal + noise
        
        data.append(noisy_signal[:context_len])
        ground_truth.append(signal[context_len:])
        
    return np.array(data), np.array(ground_truth)

def run_benchmark():
    print("Loading TimesFM 2.5 200M model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize model
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    
    model.compile(
        timesfm.ForecastConfig(
            max_context=512,
            max_horizon=128,
            normalize_inputs=True,
            use_continuous_quantile_head=False,
            force_flip_invariance=True, 
        )
    )

    print("Model loaded.")

    batch_size = 32
    context_len = 512
    horizon_len = 128
    
    print("Generating synthetic data...")
    inputs, ground_truth = generate_synthetic_data(batch_size, context_len, horizon_len)
    
    # Run warmup
    print("Running warmup...")
    # v2.5 forecast takes list of arrays and doesn't need freq for array inputs if not specified? 
    # README says "The input time series contexts... along with their frequencies."
    # But v2.5 README code example:
    # point_forecast, quantile_forecast = model.forecast(horizon=12, inputs=[...])
    # It doesn't show freq in the example for v2.5 200M.
    # Let's check configs.py or timesfm_2p5/timesfm_2p5_base.py to be sure about forecast signature.
    # The README example:
    # point_forecast, quantile_forecast = model.forecast(
    #     horizon=12,
    #     inputs=[...],
    # )
    model.forecast(horizon=horizon_len, inputs=list(inputs))
    
    # Benchmark loop
    num_runs = 10
    total_time = 0
    
    print(f"Starting benchmark ({num_runs} runs)...")
    start_time = time.time()
    
    for i in range(num_runs):
        iter_start = time.time()
        point_forecast, _ = model.forecast(horizon=horizon_len, inputs=list(inputs))
        iter_end = time.time()
        iter_time = iter_end - iter_start
        total_time += iter_time
        print(f"Run {i+1}: {iter_time:.4f}s")
        
    avg_latency = total_time / num_runs
    print(f"\nAverage Latency: {avg_latency:.4f}s per batch (batch_size={batch_size})")
    print(f"Throughput: {batch_size / avg_latency:.2f} samples/s")

    # Calculate metrics on the last run's output
    # point_forecast shape: (B, H)
    # ground_truth shape: (B, H)
    mae = np.mean(np.abs(point_forecast - ground_truth))
    mse = np.mean(np.square(point_forecast - ground_truth))
    
    print("\nAccuracy Metrics (vs Clean Signal):")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")

if __name__ == "__main__":
    run_benchmark()
